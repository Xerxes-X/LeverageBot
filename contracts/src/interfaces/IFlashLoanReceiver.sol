// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title IFlashLoanReceiver
/// @notice Interface for Aave V3 flash loan callbacks.
/// @dev The Pool contract calls executeOperation after transferring the flash-borrowed assets.
interface IFlashLoanReceiver {
    /// @notice Called by Aave Pool after flash loan assets are transferred to the receiver.
    /// @param assets The addresses of the flash-borrowed assets.
    /// @param amounts The amounts of the flash-borrowed assets.
    /// @param premiums The fees for each flash-borrowed asset.
    /// @param initiator The address that initiated the flash loan (must be this contract).
    /// @param params Arbitrary bytes forwarded from the flash loan caller.
    /// @return True if the operation was successful and the Pool should proceed with repayment.
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external returns (bool);
}
