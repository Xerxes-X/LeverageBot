//! Position lifecycle manager for BSC Leverage Bot.
//!
//! Orchestrates position actions (open, close, deleverage, increase) by
//! composing aggregator quotes, LeverageExecutor calldata, and tx submission.
//! Handles both LONG and SHORT positions using the same contract with
//! parameterized token roles.
//!
//! Transaction flow:
//! 1. Get swap quote from aggregator (parallel fan-out)
//! 2. Encode call to `LeverageExecutor.openLeveragePosition()` (or close/deleverage)
//! 3. Simulate via `eth_call`
//! 4. Submit via MEV-protected RPC
//! 5. Verify post-execution Aave state
//! 6. Record in P&L tracker

use alloy::network::TransactionBuilder;
use alloy::primitives::{Address, Bytes, U256};
use alloy::rpc::types::TransactionRequest;
use alloy::sol_types::SolCall;
use anyhow::{Context, Result};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::config::{AaveAssetConfig, AaveConfig, PositionConfig, TokenConfig};
use crate::constants::{DEFAULT_FLASH_LOAN_PREMIUM, TOKEN_WBNB};
use crate::core::safety::SafetyState;
use crate::errors::BotError;
use crate::execution::aave_client::AaveClient;
use crate::execution::aggregator_client::AggregatorClient;
use crate::execution::contracts::ILeverageExecutor;
use crate::execution::tx_submitter::TxSubmitter;
use crate::types::{PositionDirection, PositionState, SwapQuote};

use super::pnl_tracker::PnLTracker;

/// Currently active position with its database ID.
struct ActivePosition {
    state: PositionState,
    id: i64,
}

/// Direction-aware position lifecycle manager.
///
/// Handles both LONG (volatile collateral, stable debt) and SHORT
/// (stable collateral, volatile debt) positions through the same
/// `LeverageExecutor` contract with parameterized token roles.
pub struct PositionManager {
    aave_client: Arc<AaveClient>,
    aggregator_client: Arc<AggregatorClient>,
    tx_submitter: Arc<TxSubmitter>,
    pnl_tracker: Arc<PnLTracker>,
    safety: Arc<SafetyState>,
    executor_address: Address,
    user_address: Address,
    config: PositionConfig,
    aave_config: AaveConfig,
    tokens: HashMap<String, TokenConfig>,
    current_position: RwLock<Option<ActivePosition>>,
}

impl PositionManager {
    /// Construct with all required dependencies.
    pub fn new(
        aave_client: Arc<AaveClient>,
        aggregator_client: Arc<AggregatorClient>,
        tx_submitter: Arc<TxSubmitter>,
        pnl_tracker: Arc<PnLTracker>,
        safety: Arc<SafetyState>,
        executor_address: Address,
        user_address: Address,
        config: PositionConfig,
        aave_config: AaveConfig,
        tokens: HashMap<String, TokenConfig>,
    ) -> Self {
        Self {
            aave_client,
            aggregator_client,
            tx_submitter,
            pnl_tracker,
            safety,
            executor_address,
            user_address,
            config,
            aave_config,
            tokens,
            current_position: RwLock::new(None),
        }
    }

    // -----------------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------------

    /// Whether there is an open position.
    pub async fn has_open_position(&self) -> bool {
        self.current_position.read().await.is_some()
    }

    /// Get a snapshot of the current position state.
    pub async fn current_position(&self) -> Option<PositionState> {
        self.current_position
            .read()
            .await
            .as_ref()
            .map(|p| p.state.clone())
    }

    // -----------------------------------------------------------------------
    // Open position
    // -----------------------------------------------------------------------

    /// Open a new leveraged position.
    ///
    /// - **LONG**: flash loan `debt_token` (stable) → swap to `collateral_token` (volatile) → supply
    /// - **SHORT**: flash loan `debt_token` (volatile) → swap to `collateral_token` (stable) → supply
    ///
    /// `amount` is in USD terms (converted to native token units internally).
    pub async fn open_position(
        &self,
        direction: PositionDirection,
        debt_token: &str,
        amount: Decimal,
        collateral_token: &str,
    ) -> Result<PositionState, BotError> {
        // 1. Check no existing position
        if self.has_open_position().await {
            return Err(BotError::PositionError {
                reason: "cannot open: position already open".into(),
            });
        }

        // 2. Isolation mode check (SHORT only)
        let collateral_addr = self.get_token_address(collateral_token)?;
        if direction == PositionDirection::Short {
            self.check_isolation_mode(collateral_addr).await?;
            info!("isolation mode check passed for {collateral_token}");
        }

        // 3. Resolve token metadata
        let debt_addr = self.get_token_address(debt_token)?;
        let debt_decimals = self.get_token_decimals(debt_token)?;

        // Convert USD amount to native token units
        let amount_native = decimal_to_u256(amount * Decimal::from(10u64.pow(debt_decimals as u32)));

        // 4. Get best swap quote (debt → collateral)
        let quote = self
            .aggregator_client
            .get_best_quote(
                debt_addr,
                collateral_addr,
                amount_native,
                self.config.max_slippage_bps,
            )
            .await?;

        info!(
            provider = %quote.provider,
            to_amount = %quote.to_amount,
            "best quote received"
        );

        // 5. Encode LeverageExecutor.openLeveragePosition calldata
        let router_addr: Address = quote
            .router_address
            .parse()
            .map_err(|_| BotError::PositionError {
                reason: format!("invalid router address: {}", quote.router_address),
            })?;
        let swap_calldata = Bytes::from(quote.calldata.clone());
        let min_collateral_out = decimal_to_u256(quote.to_amount_min);

        let call = ILeverageExecutor::openLeveragePositionCall {
            debtAsset: debt_addr,
            flashAmount: amount_native,
            collateralAsset: collateral_addr,
            swapRouter: router_addr,
            swapCalldata: swap_calldata,
            minCollateralOut: min_collateral_out,
        };
        let calldata = Bytes::from(call.abi_encode());

        // 6. Build transaction
        let mut tx = TransactionRequest::default();
        tx.set_from(self.user_address);
        tx.set_to(self.executor_address);
        tx.set_value(U256::ZERO);
        tx.set_input(calldata);
        tx.set_gas_limit(800_000);

        // 7. Dry-run / paper trading check
        if self.safety.is_dry_run() {
            self.tx_submitter.simulate(&tx).await?;

            let position =
                self.build_approximate_state(direction, debt_token, collateral_token, amount, &quote);

            // Log comprehensive paper trade details
            info!(
                "╔══════════════════════════════════════════════════════════════════╗"
            );
            info!(
                "║                    PAPER TRADE: OPEN POSITION                    ║"
            );
            info!(
                "╠══════════════════════════════════════════════════════════════════╣"
            );
            info!(
                direction = direction.as_str(),
                debt_token = debt_token,
                collateral_token = collateral_token,
                "║ Direction & Tokens"
            );
            info!(
                amount_usd = %amount,
                quote_to_amount = %quote.to_amount,
                quote_from_amount = %quote.from_amount,
                "║ Position Size"
            );
            info!(
                provider = %quote.provider,
                router = %quote.router_address,
                to_amount_min = %quote.to_amount_min,
                "║ Swap Quote"
            );
            info!(
                health_factor = %position.health_factor,
                debt_usd = %position.debt_usd,
                collateral_usd = %position.collateral_usd,
                "║ Aave State"
            );
            info!(
                liquidation_threshold = %position.liquidation_threshold,
                borrow_rate_ray = %position.borrow_rate_ray,
                "║ Risk Parameters"
            );
            info!(
                executor = %self.executor_address,
                user = %self.user_address,
                "║ Addresses"
            );
            info!(
                "╚══════════════════════════════════════════════════════════════════╝"
            );

            let mut guard = self.current_position.write().await;
            *guard = Some(ActivePosition {
                state: position.clone(),
                id: -1,
            });
            return Ok(position);
        }

        // 8. Safety gate for position opening
        self.safety
            .can_open_position(amount, dec!(2.0))?;

        // 9. Submit and wait (includes simulate + safety gas check + submit + confirm)
        let receipt = self
            .tx_submitter
            .submit_and_wait(tx, &self.safety)
            .await?;

        let gas_used = receipt.gas_used;
        self.safety.record_action();

        // 10. Verify post-execution Aave state
        let account = self
            .aave_client
            .get_user_account_data(self.executor_address)
            .await
            .map_err(|e| BotError::AaveError {
                reason: format!("failed to get post-open account data: {e}"),
            })?;

        let reserve = self
            .aave_client
            .get_reserve_data(debt_addr)
            .await
            .map_err(|e| BotError::AaveError {
                reason: format!("failed to get post-open reserve data: {e}"),
            })?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let position = PositionState {
            direction,
            debt_token: debt_token.to_string(),
            collateral_token: collateral_token.to_string(),
            debt_usd: account.total_debt_usd,
            collateral_usd: account.total_collateral_usd,
            initial_debt_usd: amount,
            initial_collateral_usd: account.total_collateral_usd,
            health_factor: account.health_factor,
            borrow_rate_ray: reserve.variable_borrow_rate,
            liquidation_threshold: account.current_liquidation_threshold,
            open_timestamp: now,
        };

        // 11. Record to P&L tracker
        let tx_hash = format!("{:?}", receipt.transaction_hash);
        let gas_cost_usd = self.estimate_gas_cost_usd(gas_used).await.map_err(|e| BotError::AaveError {
            reason: format!("gas cost estimation failed: {e}"),
        })?;
        let position_id = self
            .pnl_tracker
            .record_open(&position, &tx_hash, gas_cost_usd)
            .await
            .map_err(|e| BotError::DatabaseError {
                reason: format!("failed to record position open: {e}"),
            })?;

        let mut guard = self.current_position.write().await;
        *guard = Some(ActivePosition {
            state: position.clone(),
            id: position_id,
        });

        info!(
            direction = direction.as_str(),
            health_factor = %account.health_factor,
            collateral = %account.total_collateral_usd,
            debt = %account.total_debt_usd,
            "position opened"
        );

        Ok(position)
    }

    // -----------------------------------------------------------------------
    // Close position
    // -----------------------------------------------------------------------

    /// Close the current position entirely.
    ///
    /// Calls `LeverageExecutor.closeLeveragePosition()` which internally
    /// flash loans the debt amount (mode=0), repays Aave, withdraws collateral,
    /// swaps back to debt token, and repays the flash loan.
    pub async fn close_position(
        &self,
        reason: &str,
    ) -> Result<PositionState, BotError> {
        let (pos, position_id) = {
            let guard = self.current_position.read().await;
            let active = guard.as_ref().ok_or(BotError::PositionError {
                reason: "no open position to close".into(),
            })?;
            (active.state.clone(), active.id)
        };

        let debt_addr = self.get_token_address(&pos.debt_token)?;
        let collateral_addr = self.get_token_address(&pos.collateral_token)?;
        let debt_decimals = self.get_token_decimals(&pos.debt_token)?;
        let collateral_decimals = self.get_token_decimals(&pos.collateral_token)?;

        // Get current on-chain debt (with accrued interest)
        let account = self
            .aave_client
            .get_user_account_data(self.executor_address)
            .await
            .map_err(|e| BotError::AaveError {
                reason: format!("failed to get account data for close: {e}"),
            })?;

        let debt_native = decimal_to_u256(
            account.total_debt_usd * Decimal::from(10u64.pow(debt_decimals as u32)),
        );
        // Add 0.1% buffer for interest accruing during tx
        let debt_with_buffer = debt_native + debt_native / U256::from(1000);

        let collateral_native = decimal_to_u256(
            account.total_collateral_usd * Decimal::from(10u64.pow(collateral_decimals as u32)),
        );

        // Get swap quote: collateral → debt
        let quote = self
            .aggregator_client
            .get_best_quote(
                collateral_addr,
                debt_addr,
                collateral_native,
                self.config.max_slippage_bps,
            )
            .await?;

        let router_addr: Address = quote
            .router_address
            .parse()
            .map_err(|_| BotError::PositionError {
                reason: format!("invalid router address: {}", quote.router_address),
            })?;

        let min_debt_out = decimal_to_u256(quote.to_amount_min);

        let call = ILeverageExecutor::closeLeveragePositionCall {
            debtAsset: debt_addr,
            debtAmount: debt_with_buffer,
            collateralAsset: collateral_addr,
            collateralToWithdraw: collateral_native,
            swapRouter: router_addr,
            swapCalldata: Bytes::from(quote.calldata.clone()),
            minDebtTokenOut: min_debt_out,
        };
        let calldata = Bytes::from(call.abi_encode());

        let mut tx = TransactionRequest::default();
        tx.set_from(self.user_address);
        tx.set_to(self.executor_address);
        tx.set_value(U256::ZERO);
        tx.set_input(calldata);
        tx.set_gas_limit(900_000);

        if self.safety.is_dry_run() {
            self.tx_submitter.simulate(&tx).await?;

            // Calculate approximate P&L
            let entry_debt = pos.initial_debt_usd;
            let exit_collateral = account.total_collateral_usd;
            let approx_pnl = exit_collateral - entry_debt;
            let pnl_pct = if entry_debt > Decimal::ZERO {
                (approx_pnl / entry_debt) * dec!(100)
            } else {
                Decimal::ZERO
            };

            // Log comprehensive paper trade close details
            info!(
                "╔══════════════════════════════════════════════════════════════════╗"
            );
            info!(
                "║                   PAPER TRADE: CLOSE POSITION                    ║"
            );
            info!(
                "╠══════════════════════════════════════════════════════════════════╣"
            );
            info!(
                direction = pos.direction.as_str(),
                debt_token = %pos.debt_token,
                collateral_token = %pos.collateral_token,
                "║ Position"
            );
            info!(
                initial_debt_usd = %pos.initial_debt_usd,
                current_debt_usd = %account.total_debt_usd,
                current_collateral_usd = %account.total_collateral_usd,
                "║ Size & Value"
            );
            info!(
                approx_pnl_usd = %approx_pnl,
                pnl_percent = %pnl_pct,
                "║ Estimated P&L"
            );
            info!(
                quote_provider = %quote.provider,
                quote_to_amount = %quote.to_amount,
                quote_from_amount = %quote.from_amount,
                "║ Exit Quote"
            );
            info!(
                "╚══════════════════════════════════════════════════════════════════╝"
            );

            let closed = PositionState {
                direction: pos.direction,
                debt_token: pos.debt_token,
                collateral_token: pos.collateral_token,
                debt_usd: Decimal::ZERO,
                collateral_usd: Decimal::ZERO,
                initial_debt_usd: pos.initial_debt_usd,
                initial_collateral_usd: pos.initial_collateral_usd,
                health_factor: Decimal::ZERO,
                borrow_rate_ray: Decimal::ZERO,
                liquidation_threshold: pos.liquidation_threshold,
                open_timestamp: pos.open_timestamp,
            };

            let mut guard = self.current_position.write().await;
            *guard = None;
            return Ok(closed);
        }

        let receipt = self
            .tx_submitter
            .submit_and_wait(tx, &self.safety)
            .await?;

        let gas_used = receipt.gas_used;
        self.safety.record_action();

        let tx_hash = format!("{:?}", receipt.transaction_hash);
        let gas_cost_usd = self.estimate_gas_cost_usd(gas_used).await.map_err(|e| BotError::AaveError {
            reason: format!("gas cost estimation failed: {e}"),
        })?;

        // Approximate tokens received (surplus after flash loan repayment)
        let tokens_received = {
            let quote_output_usd =
                quote.to_amount / Decimal::from(10u64.pow(debt_decimals as u32));
            let surplus = quote_output_usd - account.total_debt_usd;
            surplus.max(Decimal::ZERO)
        };

        // Record close in P&L tracker
        if position_id >= 0 {
            let pnl = self
                .pnl_tracker
                .record_close(position_id, &tx_hash, gas_cost_usd, tokens_received, reason)
                .await
                .map_err(|e| BotError::DatabaseError {
                    reason: format!("failed to record position close: {e}"),
                })?;
            info!(net_pnl = %pnl.net_pnl_usd, "realized P&L");
        }

        let closed = PositionState {
            direction: pos.direction,
            debt_token: pos.debt_token,
            collateral_token: pos.collateral_token,
            debt_usd: Decimal::ZERO,
            collateral_usd: Decimal::ZERO,
            initial_debt_usd: pos.initial_debt_usd,
            initial_collateral_usd: pos.initial_collateral_usd,
            health_factor: Decimal::ZERO,
            borrow_rate_ray: Decimal::ZERO,
            liquidation_threshold: pos.liquidation_threshold,
            open_timestamp: pos.open_timestamp,
        };

        let mut guard = self.current_position.write().await;
        *guard = None;

        info!(
            direction = pos.direction.as_str(),
            reason,
            "position closed"
        );
        Ok(closed)
    }

    // -----------------------------------------------------------------------
    // Partial deleverage
    // -----------------------------------------------------------------------

    /// Reduce position size to bring health factor back to `target_hf`.
    ///
    /// Uses the deleverage formula:
    ///     `repay_amount = (D - C * LT / h_t) / (1 + f - LT / h_t)`
    /// where D=debt, C=collateral, LT=liquidation threshold, h_t=target HF,
    /// f=flash loan premium.
    pub async fn partial_deleverage(
        &self,
        target_hf: Decimal,
    ) -> Result<PositionState, BotError> {
        let (pos, position_id) = {
            let guard = self.current_position.read().await;
            let active = guard.as_ref().ok_or(BotError::PositionError {
                reason: "no open position to deleverage".into(),
            })?;
            (active.state.clone(), active.id)
        };

        // Refresh on-chain data
        let account = self
            .aave_client
            .get_user_account_data(self.executor_address)
            .await
            .map_err(|e| BotError::AaveError {
                reason: format!("failed to get account data for deleverage: {e}"),
            })?;

        if account.health_factor >= target_hf {
            info!(
                current_hf = %account.health_factor,
                target_hf = %target_hf,
                "HF already above target, no deleverage needed"
            );
            return Ok(pos);
        }

        // Compute repay amount using analytical formula
        let lt = pos.liquidation_threshold;
        let debt = account.total_debt_usd;
        let collateral = account.total_collateral_usd;

        let lt_over_ht = lt / target_hf;
        let numerator = debt - collateral * lt_over_ht;
        let denominator = dec!(1) + DEFAULT_FLASH_LOAN_PREMIUM - lt_over_ht;

        if denominator <= Decimal::ZERO {
            return Err(BotError::PositionError {
                reason: format!(
                    "deleverage formula denominator <= 0: target_hf={target_hf} may be unreachable"
                ),
            });
        }

        let repay_amount_usd = numerator / denominator;
        if repay_amount_usd <= Decimal::ZERO {
            info!("computed repay amount <= 0, no deleverage needed");
            return Ok(pos);
        }

        info!(
            repay = %repay_amount_usd,
            target_hf = %target_hf,
            current_hf = %account.health_factor,
            "deleveraging position"
        );

        let debt_addr = self.get_token_address(&pos.debt_token)?;
        let collateral_addr = self.get_token_address(&pos.collateral_token)?;
        let debt_decimals = self.get_token_decimals(&pos.debt_token)?;
        let collateral_decimals = self.get_token_decimals(&pos.collateral_token)?;

        let repay_native = decimal_to_u256(
            repay_amount_usd * Decimal::from(10u64.pow(debt_decimals as u32)),
        );

        // Estimate collateral to sell
        let collateral_to_sell = if collateral > Decimal::ZERO {
            decimal_to_u256(
                (repay_amount_usd / collateral)
                    * account.total_collateral_usd
                    * Decimal::from(10u64.pow(collateral_decimals as u32)),
            )
        } else {
            U256::ZERO
        };

        // Get swap quote: collateral → debt
        let quote = self
            .aggregator_client
            .get_best_quote(
                collateral_addr,
                debt_addr,
                collateral_to_sell,
                self.config.max_slippage_bps,
            )
            .await?;

        let router_addr: Address = quote
            .router_address
            .parse()
            .map_err(|_| BotError::PositionError {
                reason: format!("invalid router address: {}", quote.router_address),
            })?;

        let call = ILeverageExecutor::deleveragePositionCall {
            debtAsset: debt_addr,
            repayAmount: repay_native,
            collateralAsset: collateral_addr,
            collateralToWithdraw: collateral_to_sell,
            swapRouter: router_addr,
            swapCalldata: Bytes::from(quote.calldata.clone()),
            minDebtTokenOut: decimal_to_u256(quote.to_amount_min),
        };
        let calldata = Bytes::from(call.abi_encode());

        let mut tx = TransactionRequest::default();
        tx.set_from(self.user_address);
        tx.set_to(self.executor_address);
        tx.set_value(U256::ZERO);
        tx.set_input(calldata);
        tx.set_gas_limit(900_000);

        if self.safety.is_dry_run() {
            self.tx_submitter.simulate(&tx).await?;

            // Estimate new HF after deleverage
            let new_debt = debt - repay_amount_usd;
            let new_collateral = collateral - repay_amount_usd;
            let new_hf = if new_debt > Decimal::ZERO {
                (new_collateral * lt) / new_debt
            } else {
                dec!(999)
            };

            info!(
                "╔══════════════════════════════════════════════════════════════════╗"
            );
            info!(
                "║                  PAPER TRADE: DELEVERAGE POSITION                ║"
            );
            info!(
                "╠══════════════════════════════════════════════════════════════════╣"
            );
            info!(
                direction = pos.direction.as_str(),
                current_hf = %account.health_factor,
                target_hf = %target_hf,
                "║ Health Factor"
            );
            info!(
                repay_usd = %repay_amount_usd,
                collateral_sold_usd = %repay_amount_usd,
                "║ Deleverage Amount"
            );
            info!(
                debt_before = %debt,
                debt_after = %new_debt,
                collateral_before = %collateral,
                collateral_after = %new_collateral,
                "║ Position Change"
            );
            info!(
                estimated_new_hf = %new_hf,
                "║ Post-Deleverage HF"
            );
            info!(
                "╚══════════════════════════════════════════════════════════════════╝"
            );
            return Ok(pos);
        }

        let receipt = self
            .tx_submitter
            .submit_and_wait(tx, &self.safety)
            .await?;

        let gas_used = receipt.gas_used;
        self.safety.record_action();

        let tx_hash = format!("{:?}", receipt.transaction_hash);
        let gas_cost_usd = self.estimate_gas_cost_usd(gas_used).await.map_err(|e| BotError::AaveError {
            reason: format!("gas cost estimation failed: {e}"),
        })?;

        if position_id >= 0 {
            self.pnl_tracker
                .record_deleverage(position_id, &tx_hash, gas_cost_usd)
                .await
                .map_err(|e| BotError::DatabaseError {
                    reason: format!("failed to record deleverage: {e}"),
                })?;
        }

        // Refresh state
        let account = self
            .aave_client
            .get_user_account_data(self.executor_address)
            .await
            .map_err(|e| BotError::AaveError {
                reason: format!("failed to get post-deleverage account data: {e}"),
            })?;
        let reserve = self
            .aave_client
            .get_reserve_data(debt_addr)
            .await
            .map_err(|e| BotError::AaveError {
                reason: format!("failed to get post-deleverage reserve data: {e}"),
            })?;

        let updated = PositionState {
            direction: pos.direction,
            debt_token: pos.debt_token,
            collateral_token: pos.collateral_token,
            debt_usd: account.total_debt_usd,
            collateral_usd: account.total_collateral_usd,
            initial_debt_usd: pos.initial_debt_usd,
            initial_collateral_usd: pos.initial_collateral_usd,
            health_factor: account.health_factor,
            borrow_rate_ray: reserve.variable_borrow_rate,
            liquidation_threshold: account.current_liquidation_threshold,
            open_timestamp: pos.open_timestamp,
        };

        let mut guard = self.current_position.write().await;
        if let Some(ref mut active) = *guard {
            active.state = updated.clone();
        }

        info!(
            new_hf = %account.health_factor,
            debt = %account.total_debt_usd,
            collateral = %account.total_collateral_usd,
            "deleverage complete"
        );

        Ok(updated)
    }

    // -----------------------------------------------------------------------
    // State refresh
    // -----------------------------------------------------------------------

    /// Refresh the current position's on-chain state from Aave.
    pub async fn refresh_position(&self) -> Result<Option<PositionState>> {
        let pos = match self.current_position().await {
            Some(p) => p,
            None => return Ok(None),
        };

        let debt_addr = self.get_token_address(&pos.debt_token)?;

        let account = self
            .aave_client
            .get_user_account_data(self.executor_address)
            .await
            .context("failed to refresh account data")?;
        let reserve = self
            .aave_client
            .get_reserve_data(debt_addr)
            .await
            .context("failed to refresh reserve data")?;

        let updated = PositionState {
            direction: pos.direction,
            debt_token: pos.debt_token,
            collateral_token: pos.collateral_token,
            debt_usd: account.total_debt_usd,
            collateral_usd: account.total_collateral_usd,
            initial_debt_usd: pos.initial_debt_usd,
            initial_collateral_usd: pos.initial_collateral_usd,
            health_factor: account.health_factor,
            borrow_rate_ray: reserve.variable_borrow_rate,
            liquidation_threshold: account.current_liquidation_threshold,
            open_timestamp: pos.open_timestamp,
        };

        let mut guard = self.current_position.write().await;
        if let Some(ref mut active) = *guard {
            active.state = updated.clone();
        }

        Ok(Some(updated))
    }

    // -----------------------------------------------------------------------
    // Isolation mode check
    // -----------------------------------------------------------------------

    /// Check if collateral asset is restricted by Aave V3 Isolation Mode.
    ///
    /// Rejects if the asset's isolated debt ceiling is fully utilized.
    async fn check_isolation_mode(&self, collateral_addr: Address) -> Result<(), BotError> {
        let reserve = self
            .aave_client
            .get_reserve_data(collateral_addr)
            .await
            .map_err(|e| BotError::PositionError {
                reason: format!("failed to check isolation mode: {e}"),
            })?;

        if reserve.isolation_mode_enabled
            && reserve.debt_ceiling > Decimal::ZERO
            && reserve.current_isolated_debt >= reserve.debt_ceiling
        {
            return Err(BotError::PositionError {
                reason: format!(
                    "isolation debt ceiling reached: {} >= {}",
                    reserve.current_isolated_debt, reserve.debt_ceiling
                ),
            });
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Resolve token symbol to checksummed address.
    fn get_token_address(&self, symbol: &str) -> Result<Address, BotError> {
        let token = self.tokens.get(symbol).ok_or(BotError::PositionError {
            reason: format!("unknown token symbol: {symbol}"),
        })?;
        token
            .address
            .parse()
            .map_err(|_| BotError::PositionError {
                reason: format!("invalid token address for {symbol}: {}", token.address),
            })
    }

    /// Resolve token symbol to decimal count.
    fn get_token_decimals(&self, symbol: &str) -> Result<u8, BotError> {
        let token = self.tokens.get(symbol).ok_or(BotError::PositionError {
            reason: format!("unknown token symbol: {symbol}"),
        })?;
        Ok(token.decimals)
    }

    /// Build an approximate PositionState for dry-run mode.
    fn build_approximate_state(
        &self,
        direction: PositionDirection,
        debt_token: &str,
        collateral_token: &str,
        amount: Decimal,
        quote: &SwapQuote,
    ) -> PositionState {
        let collateral_decimals = self
            .get_token_decimals(collateral_token)
            .unwrap_or(18);
        let collateral_usd =
            quote.to_amount / Decimal::from(10u64.pow(collateral_decimals as u32));

        // Approximate LT from config
        let lt = self
            .aave_config
            .supported_assets
            .get(collateral_token)
            .map(|a| Decimal::from(a.liquidation_threshold_bps) / dec!(10_000))
            .unwrap_or(dec!(0.80));

        let hf = if amount > Decimal::ZERO {
            (collateral_usd * lt) / amount
        } else {
            Decimal::ZERO
        };

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        PositionState {
            direction,
            debt_token: debt_token.to_string(),
            collateral_token: collateral_token.to_string(),
            debt_usd: amount,
            collateral_usd,
            initial_debt_usd: amount,
            initial_collateral_usd: collateral_usd,
            health_factor: hf,
            borrow_rate_ray: Decimal::ZERO,
            liquidation_threshold: lt,
            open_timestamp: now,
        }
    }

    /// Estimate gas cost in USD from gas units used.
    async fn estimate_gas_cost_usd(&self, gas_used: u64) -> Result<Decimal> {
        if gas_used == 0 {
            return Ok(Decimal::ZERO);
        }
        let (gas_price, _) = self.tx_submitter.get_gas_price().await?;
        let gas_cost_wei = gas_used as u128 * gas_price;
        let gas_cost_bnb = Decimal::from(gas_cost_wei) / crate::constants::WAD;
        let bnb_price = self
            .aave_client
            .get_asset_price(TOKEN_WBNB)
            .await
            .context("failed to get BNB price for gas cost estimation")?;
        Ok(gas_cost_bnb * bnb_price)
    }
}

/// Convert a Decimal to U256 for on-chain token amounts.
///
/// Truncates any fractional part (token amounts are integer native units).
fn decimal_to_u256(d: Decimal) -> U256 {
    U256::from(d.trunc().to_u128().unwrap_or(0))
}
