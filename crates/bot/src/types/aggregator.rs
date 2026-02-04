use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Best swap quote selected from parallel fan-out across DEX aggregators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapQuote {
    /// Provider name: "1inch", "openocean", "paraswap".
    pub provider: String,
    pub from_token: String,
    pub to_token: String,
    #[serde(with = "rust_decimal::serde::str")]
    pub from_amount: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub to_amount: Decimal,
    #[serde(with = "rust_decimal::serde::str")]
    pub to_amount_min: Decimal,
    /// Raw calldata for the aggregator router.
    #[serde(with = "hex_bytes")]
    pub calldata: Vec<u8>,
    /// Router contract address to call with `calldata`.
    pub router_address: String,
    pub gas_estimate: u64,
    #[serde(with = "rust_decimal::serde::str")]
    pub price_impact: Decimal,
}

/// Hex-encode/decode Vec<u8> for JSON serialization.
mod hex_bytes {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex_string = format!("0x{}", hex::encode(bytes));
        serializer.serialize_str(&hex_string)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let s = s.strip_prefix("0x").unwrap_or(&s);
        hex::decode(s).map_err(serde::de::Error::custom)
    }
}
