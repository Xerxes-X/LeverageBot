// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title IAaveV3Pool
/// @notice Minimal interface for Aave V3 Pool methods used by LeverageExecutor.
/// @dev Only declares the subset of IPool functions we actually call.
interface IAaveV3Pool {
    /// @notice Executes a flash loan for multiple assets.
    /// @param receiverAddress The contract implementing IFlashLoanReceiver.
    /// @param assets Token addresses to borrow.
    /// @param amounts Amounts to borrow (in token's native decimals).
    /// @param interestRateModes 0 = repay in same tx, 2 = open variable debt.
    /// @param onBehalfOf Address that will receive the debt (if mode != 0).
    /// @param params Arbitrary bytes forwarded to executeOperation.
    /// @param referralCode Referral code (0 for none).
    function flashLoan(
        address receiverAddress,
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata interestRateModes,
        address onBehalfOf,
        bytes calldata params,
        uint16 referralCode
    ) external;

    /// @notice Supplies an asset to the pool as collateral.
    /// @param asset The address of the underlying asset.
    /// @param amount The amount to supply (in native decimals).
    /// @param onBehalfOf The address that receives the aToken.
    /// @param referralCode Referral code.
    function supply(address asset, uint256 amount, address onBehalfOf, uint16 referralCode) external;

    /// @notice Withdraws an asset from the pool.
    /// @param asset The address of the underlying asset.
    /// @param amount The amount to withdraw (type(uint256).max for full withdrawal).
    /// @param to The address that receives the underlying.
    /// @return The actual amount withdrawn.
    function withdraw(address asset, uint256 amount, address to) external returns (uint256);

    /// @notice Repays a borrowed asset.
    /// @param asset The address of the borrowed asset.
    /// @param amount The amount to repay (type(uint256).max for full repayment).
    /// @param interestRateMode 2 for variable.
    /// @param onBehalfOf The address of the user with the debt.
    /// @return The actual amount repaid.
    function repay(address asset, uint256 amount, uint256 interestRateMode, address onBehalfOf)
        external
        returns (uint256);
}
