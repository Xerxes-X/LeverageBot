//! Transaction submitter — signing, simulation, MEV-protected submission.
//!
//! Ported from Python `execution/tx_submitter.py`. Signs transactions locally,
//! simulates via `eth_call`, submits via MEV-protected RPC (48 Club Privacy),
//! and manages nonce state with async-safe locking.
//!
//! Provider split:
//! - **Standard RPC**: simulation (`eth_call`), gas estimation, nonce queries.
//! - **MEV RPC** (rpc.48.club): `send_raw_transaction` only — prevents
//!   sandwich attacks by keeping transactions out of the public mempool.

use alloy::consensus::{SignableTransaction, TxEnvelope, TxLegacy};
use alloy::eips::eip2718::Encodable2718;
use alloy::primitives::{Bytes, TxKind, B256, U256};
use alloy::rpc::types::TransactionRequest;
use alloy::providers::Provider;
use alloy::signers::local::PrivateKeySigner;
use alloy::signers::SignerSync;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::config::TransactionTiming;
use crate::core::safety::SafetyState;
use crate::errors::BotError;
use crate::execution::aave_client::HttpProvider;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// `Error(string)` selector: `keccak256("Error(string)")[0..4]`.
const ERROR_SELECTOR: [u8; 4] = [0x08, 0xc3, 0x79, 0xa0];

/// `Panic(uint256)` selector: `keccak256("Panic(uint256)")[0..4]`.
const PANIC_SELECTOR: [u8; 4] = [0x4e, 0x48, 0x7b, 0x71];

/// Gas price safety buffer (10% above current base price).
const GAS_PRICE_BUFFER: f64 = 1.1;

/// Default gas bump for stuck transaction replacement (12.5%).
const DEFAULT_GAS_BUMP_PCT: f64 = 12.5;

/// Gas limit for a simple BNB transfer.
const SIMPLE_TRANSFER_GAS: u64 = 21_000;

// ---------------------------------------------------------------------------
// TxSubmitter
// ---------------------------------------------------------------------------

/// Transaction submitter with nonce management and MEV protection.
///
/// Uses a separate MEV-protected RPC for submission to avoid sandwich
/// attacks, while the standard RPC handles simulation and reads.
pub struct TxSubmitter {
    /// Standard RPC provider for reads, simulation, and gas estimation.
    provider: HttpProvider,
    /// MEV-protected RPC provider (48 Club) for transaction submission.
    mev_provider: HttpProvider,
    /// Local signer for transaction signing.
    signer: PrivateKeySigner,
    /// Async-safe nonce counter. `None` until first chain query.
    nonce: Mutex<Option<u64>>,
    /// Timeout for `eth_call` simulation.
    simulation_timeout: Duration,
    /// Timeout for waiting for transaction confirmation.
    confirmation_timeout: Duration,
    /// Chain ID (BSC mainnet = 56).
    chain_id: u64,
}

impl TxSubmitter {
    /// Construct from pre-built providers and config.
    ///
    /// The caller is responsible for creating the standard and MEV providers
    /// via `ProviderBuilder::new().on_http(url)`.
    pub fn new(
        provider: HttpProvider,
        mev_provider: HttpProvider,
        signer: PrivateKeySigner,
        timing: &TransactionTiming,
        chain_id: u64,
    ) -> Self {
        info!(
            address = %signer.address(),
            chain_id,
            simulation_timeout = timing.simulation_timeout_seconds,
            confirmation_timeout = timing.confirmation_timeout_seconds,
            "TxSubmitter initialized"
        );

        Self {
            provider,
            mev_provider,
            signer,
            nonce: Mutex::new(None),
            simulation_timeout: Duration::from_secs(timing.simulation_timeout_seconds),
            confirmation_timeout: Duration::from_secs(timing.confirmation_timeout_seconds),
            chain_id,
        }
    }

    /// Returns the address associated with the signer.
    pub fn signer_address(&self) -> alloy::primitives::Address {
        self.signer.address()
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Simulate a transaction via `eth_call` with timeout protection.
    ///
    /// Returns raw output bytes on success. Raises `BotError::SimulationFailed`
    /// on revert or timeout.
    pub async fn simulate(&self, tx: &TransactionRequest) -> Result<Bytes, BotError> {
        let provider = &self.provider;
        let tx_clone = tx.clone();
        let timeout_dur = self.simulation_timeout;
        match tokio::time::timeout(
            timeout_dur,
            async move { provider.call(tx_clone).await },
        )
        .await
        {
            Ok(Ok(result)) => {
                debug!(output_len = result.len(), "simulation succeeded");
                Ok(result)
            }
            Ok(Err(e)) => Err(BotError::SimulationFailed {
                reason: format!("simulation reverted: {e}"),
            }),
            Err(_) => Err(BotError::SimulationFailed {
                reason: format!(
                    "simulation timed out after {}s",
                    self.simulation_timeout.as_secs()
                ),
            }),
        }
    }

    /// Full submission flow: simulate → safety gate → submit → wait.
    ///
    /// Returns the confirmed transaction receipt.
    pub async fn submit_and_wait(
        &self,
        tx: TransactionRequest,
        safety: &SafetyState,
    ) -> Result<alloy::rpc::types::TransactionReceipt, BotError> {
        // 1. Simulate via eth_call
        self.simulate(&tx).await?;

        // 2. Safety gate — check gas price
        let (gas_price, _) = self.get_gas_price().await?;
        let gas_price_gwei = (gas_price / 1_000_000_000) as u64;
        safety.can_submit_tx(gas_price_gwei)?;

        // 3. Submit (sign + send via MEV RPC)
        let tx_hash = self.submit(tx).await?;

        // 4. Wait for on-chain confirmation
        self.wait_for_receipt(tx_hash).await
    }

    /// Sign and submit a transaction via MEV-protected RPC.
    ///
    /// Assigns nonce and gas price automatically. Returns the transaction hash.
    pub async fn submit(&self, tx: TransactionRequest) -> Result<B256, BotError> {
        let nonce = self.get_next_nonce().await?;
        let (gas_price, _) = self.get_gas_price().await?;

        // Estimate gas if not provided by caller.
        // Cast to u64: Alloy may return u128 but gas limits always fit in u64.
        #[allow(clippy::unnecessary_cast)]
        let gas_limit: u64 = match tx.gas {
            Some(gas) => gas as u64,
            None => {
                let estimate = self
                    .provider
                    .estimate_gas(tx.clone())
                    .await
                    .map_err(|e| BotError::SimulationFailed {
                        reason: format!("gas estimation failed: {e}"),
                    })?;
                estimate as u64
            }
        };

        // Extract fields from TransactionRequest
        let to = tx.to.unwrap_or(TxKind::Create);
        let value = tx.value.unwrap_or_default();
        let input = tx.input.into_input().unwrap_or_default();

        // Build, sign, and send
        let tx_hash =
            self.sign_and_send(nonce, gas_price, gas_limit, to, value, input)
                .await?;

        info!(
            tx_hash = %tx_hash,
            nonce,
            gas_price,
            gas_limit,
            "transaction submitted via MEV-protected RPC"
        );

        Ok(tx_hash)
    }

    /// Poll for a transaction receipt until confirmed or timeout.
    ///
    /// Returns `BotError::TxReverted` if the receipt has `status == 0`.
    /// Returns `BotError::TxTimeout` if confirmation takes too long.
    pub async fn wait_for_receipt(
        &self,
        tx_hash: B256,
    ) -> Result<alloy::rpc::types::TransactionReceipt, BotError> {
        let start = tokio::time::Instant::now();

        loop {
            match self.provider.get_transaction_receipt(tx_hash).await {
                Ok(Some(receipt)) => {
                    if !receipt.status() {
                        return Err(BotError::TxReverted {
                            tx_hash: tx_hash.to_string(),
                            reason: "transaction reverted on-chain".into(),
                        });
                    }
                    info!(
                        tx_hash = %tx_hash,
                        gas_used = receipt.gas_used,
                        "transaction confirmed"
                    );
                    return Ok(receipt);
                }
                Ok(None) => {
                    // Not yet mined — continue polling
                }
                Err(e) => {
                    warn!(error = %e, tx_hash = %tx_hash, "receipt poll error, retrying");
                }
            }

            if start.elapsed() >= self.confirmation_timeout {
                return Err(BotError::TxTimeout {
                    tx_hash: tx_hash.to_string(),
                    timeout_seconds: self.confirmation_timeout.as_secs(),
                });
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    /// Get current gas price as `(gas_price, priority_fee)` in Wei.
    ///
    /// Applies a 10% buffer to the base gas price. For BSC legacy
    /// transactions, only `gas_price` is used; `priority_fee` is provided
    /// for informational / logging purposes.
    pub async fn get_gas_price(&self) -> Result<(u128, u128), BotError> {
        let base_price = self.provider.get_gas_price().await?;
        let gas_price = (base_price as f64 * GAS_PRICE_BUFFER) as u128;
        let priority_fee = std::cmp::max((base_price as f64 * 0.1) as u128, 1);
        let gas_price = std::cmp::max(gas_price, priority_fee);
        Ok((gas_price, priority_fee))
    }

    /// Replace a stuck transaction by sending a zero-value self-transfer at
    /// the same nonce with 12.5% higher gas price.
    pub async fn replace_stuck_tx(&self, nonce: u64) -> Result<B256, BotError> {
        let (gas_price, _) = self.get_gas_price().await?;
        let bumped_gas = (gas_price as f64 * (1.0 + DEFAULT_GAS_BUMP_PCT / 100.0)) as u128;

        let address = self.signer.address();
        let tx_hash = self
            .sign_and_send(
                nonce,
                bumped_gas,
                SIMPLE_TRANSFER_GAS,
                TxKind::Call(address),
                U256::ZERO,
                Bytes::new(),
            )
            .await?;

        warn!(
            nonce,
            new_hash = %tx_hash,
            bumped_gas,
            "stuck transaction replaced"
        );

        Ok(tx_hash)
    }

    /// Re-sync local nonce counter from on-chain pending state.
    pub async fn recover_nonce(&self) -> Result<(), BotError> {
        let mut guard = self.nonce.lock().await;
        let chain_nonce = self
            .provider
            .get_transaction_count(self.signer.address())
            .await?;
        let old = *guard;
        *guard = Some(chain_nonce);
        warn!(old_nonce = ?old, chain_nonce, "nonce recovered from chain");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Revert decoding
    // -----------------------------------------------------------------------

    /// Decode a Solidity revert reason from raw return data.
    ///
    /// Handles:
    /// - `Error(string)` (0x08c379a0) — standard revert messages.
    /// - `Panic(uint256)` (0x4e487b71) — arithmetic and assertion panics.
    /// - Unknown selectors — falls back to hex encoding.
    pub fn decode_revert_reason(data: &[u8]) -> String {
        if data.is_empty() {
            return "Unknown revert".into();
        }

        if data.len() < 4 {
            return hex::encode(data);
        }

        // Error(string): selector(4) + offset(32) + length(32) + data
        if data[..4] == ERROR_SELECTOR && data.len() >= 68 {
            if let Ok(len_bytes) = <[u8; 8]>::try_from(&data[60..68]) {
                let str_len = u64::from_be_bytes(len_bytes) as usize;
                if data.len() >= 68 + str_len {
                    return String::from_utf8_lossy(&data[68..68 + str_len]).into_owned();
                }
            }
        }

        // Panic(uint256): selector(4) + code(32)
        if data[..4] == PANIC_SELECTOR && data.len() >= 36 {
            let code = U256::from_be_slice(&data[4..36]);
            return match code.to::<u64>() {
                0x01 => "Panic: assertion failed".into(),
                0x11 => "Panic: arithmetic overflow/underflow".into(),
                0x12 => "Panic: division by zero".into(),
                0x21 => "Panic: enum conversion out of range".into(),
                0x22 => "Panic: incorrectly encoded storage byte array".into(),
                0x31 => "Panic: pop on empty array".into(),
                0x32 => "Panic: array index out of bounds".into(),
                0x41 => "Panic: too much memory allocated".into(),
                0x51 => "Panic: called zero-initialized function pointer".into(),
                _ => format!("Panic(0x{code:x})"),
            };
        }

        hex::encode(data)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Get the next nonce, initialising from chain on the first call.
    async fn get_next_nonce(&self) -> Result<u64, BotError> {
        let mut guard = self.nonce.lock().await;
        let nonce = match *guard {
            Some(n) => n,
            None => {
                let n = self
                    .provider
                    .get_transaction_count(self.signer.address())
                    .await?;
                info!(nonce = n, "nonce initialized from chain");
                n
            }
        };
        *guard = Some(nonce + 1);
        Ok(nonce)
    }

    /// Build a legacy transaction, sign it locally, and submit raw bytes
    /// via the MEV-protected provider.
    async fn sign_and_send(
        &self,
        nonce: u64,
        gas_price: u128,
        gas_limit: u64,
        to: TxKind,
        value: U256,
        input: Bytes,
    ) -> Result<B256, BotError> {
        let tx = TxLegacy {
            chain_id: Some(self.chain_id),
            nonce,
            gas_price,
            gas_limit,
            to,
            value,
            input,
        };

        // Sign the transaction hash locally
        let sig_hash = tx.signature_hash();
        let sig = self
            .signer
            .sign_hash_sync(&sig_hash)
            .map_err(|e| BotError::SimulationFailed {
                reason: format!("transaction signing failed: {e}"),
            })?;

        // Encode as RLP-wrapped legacy envelope
        let signed = tx.into_signed(sig);
        let envelope = TxEnvelope::Legacy(signed);
        let raw = envelope.encoded_2718();

        // Submit via MEV-protected RPC
        let pending = self
            .mev_provider
            .send_raw_transaction(&raw)
            .await?;

        Ok(*pending.tx_hash())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- decode_revert_reason -----------------------------------------------

    #[test]
    fn decode_empty_data_returns_unknown() {
        assert_eq!(TxSubmitter::decode_revert_reason(&[]), "Unknown revert");
    }

    #[test]
    fn decode_short_data_returns_hex() {
        assert_eq!(TxSubmitter::decode_revert_reason(&[0xAB, 0xCD]), "abcd");
    }

    #[test]
    fn decode_error_string() {
        let msg = b"Insufficient balance";
        let mut data = Vec::with_capacity(68 + msg.len());
        // Selector
        data.extend_from_slice(&ERROR_SELECTOR);
        // Offset to string data (0x20 = 32)
        data.extend_from_slice(&[0u8; 31]);
        data.push(0x20);
        // String length
        data.extend_from_slice(&[0u8; 31]);
        data.push(msg.len() as u8);
        // String bytes
        data.extend_from_slice(msg);

        assert_eq!(
            TxSubmitter::decode_revert_reason(&data),
            "Insufficient balance"
        );
    }

    #[test]
    fn decode_error_string_utf8_lossy() {
        // Non-UTF-8 bytes should be replaced, not panic
        let mut data = Vec::with_capacity(72);
        data.extend_from_slice(&ERROR_SELECTOR);
        data.extend_from_slice(&[0u8; 31]);
        data.push(0x20);
        data.extend_from_slice(&[0u8; 31]);
        data.push(4);
        data.extend_from_slice(&[0xFF, 0xFE, 0x41, 0x42]); // invalid UTF-8 + "AB"

        let result = TxSubmitter::decode_revert_reason(&data);
        assert!(result.contains("AB"));
    }

    #[test]
    fn decode_panic_assertion() {
        let mut data = vec![0u8; 36];
        data[..4].copy_from_slice(&PANIC_SELECTOR);
        data[35] = 0x01;
        assert_eq!(
            TxSubmitter::decode_revert_reason(&data),
            "Panic: assertion failed"
        );
    }

    #[test]
    fn decode_panic_arithmetic_overflow() {
        let mut data = vec![0u8; 36];
        data[..4].copy_from_slice(&PANIC_SELECTOR);
        data[35] = 0x11;
        assert_eq!(
            TxSubmitter::decode_revert_reason(&data),
            "Panic: arithmetic overflow/underflow"
        );
    }

    #[test]
    fn decode_panic_division_by_zero() {
        let mut data = vec![0u8; 36];
        data[..4].copy_from_slice(&PANIC_SELECTOR);
        data[35] = 0x12;
        assert_eq!(
            TxSubmitter::decode_revert_reason(&data),
            "Panic: division by zero"
        );
    }

    #[test]
    fn decode_panic_array_oob() {
        let mut data = vec![0u8; 36];
        data[..4].copy_from_slice(&PANIC_SELECTOR);
        data[35] = 0x32;
        assert_eq!(
            TxSubmitter::decode_revert_reason(&data),
            "Panic: array index out of bounds"
        );
    }

    #[test]
    fn decode_unknown_selector_returns_hex() {
        let data = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
        assert_eq!(
            TxSubmitter::decode_revert_reason(&data),
            "deadbeef01020304"
        );
    }

    // -- gas constants ------------------------------------------------------

    #[test]
    fn gas_price_buffer_is_ten_percent() {
        assert!((GAS_PRICE_BUFFER - 1.1).abs() < f64::EPSILON);
    }

    #[test]
    fn gas_bump_twelve_point_five_percent() {
        let base = 1_000_000_000u128; // 1 gwei
        let bumped = (base as f64 * (1.0 + DEFAULT_GAS_BUMP_PCT / 100.0)) as u128;
        assert_eq!(bumped, 1_125_000_000); // 1.125 gwei
    }

    #[test]
    fn simple_transfer_gas_is_21k() {
        assert_eq!(SIMPLE_TRANSFER_GAS, 21_000);
    }
}
