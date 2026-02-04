//! Compile-time ABI definitions for on-chain contracts via Alloy `sol!`.
//!
//! Replaces Python's runtime JSON ABI parsing — encoding errors become
//! compile errors.

#![allow(clippy::too_many_arguments)]

use alloy::sol;

// ---------------------------------------------------------------------------
// Aave V3 Pool
// ---------------------------------------------------------------------------

sol! {
    /// Aave V3 Pool contract — core lending/borrowing entry point.
    #[sol(rpc)]
    interface IPool {
        /// Execute flash loan(s).
        function flashLoan(
            address receiverAddress,
            address[] calldata assets,
            uint256[] calldata amounts,
            uint256[] calldata interestRateModes,
            address onBehalfOf,
            bytes calldata params,
            uint16 referralCode
        ) external;

        /// Get aggregated user position data.
        function getUserAccountData(address user) external view returns (
            uint256 totalCollateralBase,
            uint256 totalDebtBase,
            uint256 availableBorrowsBase,
            uint256 currentLiquidationThreshold,
            uint256 ltv,
            uint256 healthFactor
        );

        /// Supply (deposit) asset as collateral.
        function supply(
            address asset,
            uint256 amount,
            address onBehalfOf,
            uint16 referralCode
        ) external;

        /// Withdraw collateral.
        function withdraw(
            address asset,
            uint256 amount,
            address to
        ) external returns (uint256);

        /// Repay borrowed asset.
        function repay(
            address asset,
            uint256 amount,
            uint256 interestRateMode,
            address onBehalfOf
        ) external returns (uint256);

        /// Get full reserve data (15 fields) including configuration bitmap.
        function getReserveData(address asset) external view returns (
            uint256 configuration,
            uint128 liquidityIndex,
            uint128 currentLiquidityRate,
            uint128 variableBorrowIndex,
            uint128 currentVariableBorrowRate,
            uint128 currentStableBorrowRate,
            uint40 lastUpdateTimestamp,
            uint16 id,
            address aTokenAddress,
            address stableDebtTokenAddress,
            address variableDebtTokenAddress,
            address interestRateStrategyAddress,
            uint128 accruedToTreasury,
            uint128 unbacked,
            uint128 isolationModeTotalDebt
        );

        /// Flash loan premium in basis points.
        function FLASHLOAN_PREMIUM_TOTAL() external view returns (uint128);
    }
}

// ---------------------------------------------------------------------------
// Aave V3 Pool Data Provider
// ---------------------------------------------------------------------------

sol! {
    /// Aave V3 PoolDataProvider — read-only reserve/user data.
    #[sol(rpc)]
    interface IPoolDataProvider {
        /// Get reserve data (12 flat values).
        function getReserveData(address asset) external view returns (
            uint256 unbacked,
            uint256 accruedToTreasuryScaled,
            uint256 totalAToken,
            uint256 totalStableDebt,
            uint256 totalVariableDebt,
            uint256 liquidityRate,
            uint256 variableBorrowRate,
            uint256 stableBorrowRate,
            uint256 averageStableBorrowRate,
            uint256 liquidityIndex,
            uint256 variableBorrowIndex,
            uint40 lastUpdateTimestamp
        );

        /// Get reserve configuration parameters.
        function getReserveConfigurationData(address asset) external view returns (
            uint256 decimals,
            uint256 ltv,
            uint256 liquidationThreshold,
            uint256 liquidationBonus,
            uint256 reserveFactor,
            bool usageAsCollateralEnabled,
            bool borrowingEnabled,
            bool stableBorrowRateEnabled,
            bool isActive,
            bool isFrozen
        );
    }
}

// ---------------------------------------------------------------------------
// Aave V3 Oracle
// ---------------------------------------------------------------------------

sol! {
    /// Aave V3 Oracle — wraps Chainlink feeds, returns prices in base currency (USD 8 decimals).
    #[sol(rpc)]
    interface IAaveOracle {
        function getAssetPrice(address asset) external view returns (uint256 price);
        function getAssetsPrices(address[] calldata assets) external view returns (uint256[] memory prices);
    }
}

// ---------------------------------------------------------------------------
// Chainlink Aggregator V3
// ---------------------------------------------------------------------------

sol! {
    /// Chainlink price feed interface — used for oracle freshness validation.
    #[sol(rpc)]
    interface IAggregatorV3 {
        function latestRoundData() external view returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );

        function decimals() external view returns (uint8);
    }
}
