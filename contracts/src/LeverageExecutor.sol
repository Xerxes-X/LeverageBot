// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

import {IFlashLoanReceiver} from "./interfaces/IFlashLoanReceiver.sol";
import {IAaveV3Pool} from "./interfaces/IAaveV3Pool.sol";

/// @title LeverageExecutor
/// @notice Atomic execution layer for Aave V3 flash loan leverage positions on BSC.
/// @dev Direction-agnostic: the same functions handle LONG and SHORT positions.
///      LONG:  debtAsset=USDT, collateralAsset=WBNB (borrow stable, supply volatile)
///      SHORT: debtAsset=WBNB, collateralAsset=USDT (borrow volatile, supply stable)
contract LeverageExecutor is IFlashLoanReceiver, Ownable, Pausable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // =========================================================================
    //                              CONSTANTS
    // =========================================================================

    uint256 private constant FLASH_LOAN_MODE_NO_DEBT = 0;
    uint256 private constant FLASH_LOAN_MODE_VARIABLE_DEBT = 2;
    uint256 private constant VARIABLE_RATE_MODE = 2;

    uint8 private constant OP_OPEN = 0;
    uint8 private constant OP_CLOSE = 1;
    uint8 private constant OP_DELEVERAGE = 2;

    // =========================================================================
    //                           STATE VARIABLES
    // =========================================================================

    IAaveV3Pool public immutable AAVE_POOL;
    mapping(address => bool) public approvedRouters;

    /// @dev Packed params for close/deleverage to avoid stack-too-deep.
    struct CloseParams {
        address swapRouter;
        bytes swapCalldata;
        address collateralAsset;
        uint256 collateralToWithdraw;
        uint256 minDebtTokenOut;
    }

    // =========================================================================
    //                              EVENTS
    // =========================================================================

    event PositionOpened(
        address indexed debtAsset, uint256 flashAmount, address indexed collateralAsset, uint256 collateralReceived
    );

    event PositionClosed(
        address indexed debtAsset, uint256 debtRepaid, address indexed collateralAsset, uint256 collateralWithdrawn
    );

    event PositionDeleveraged(
        address indexed debtAsset, uint256 debtRepaid, address indexed collateralAsset, uint256 collateralWithdrawn
    );

    event RouterApprovalSet(address indexed router, bool approved);

    event TokensRescued(address indexed token, uint256 amount, address indexed to);

    // =========================================================================
    //                            CUSTOM ERRORS
    // =========================================================================

    error InvalidCaller(address expected, address actual);
    error InvalidInitiator(address expected, address actual);
    error RouterNotApproved(address router);
    error SwapFailed(address router);
    error SlippageExceeded(uint256 received, uint256 minimum);
    error InvalidOperationType(uint8 opType);
    error ZeroAddress();
    error ZeroAmount();

    // =========================================================================
    //                            CONSTRUCTOR
    // =========================================================================

    /// @param aavePool The Aave V3 Pool address (immutable after deployment).
    /// @param initialOwner The address that will own this contract (bot operator).
    /// @param routers Initial list of approved DEX aggregator routers.
    constructor(address aavePool, address initialOwner, address[] memory routers) Ownable(initialOwner) {
        if (aavePool == address(0)) revert ZeroAddress();
        // initialOwner zero-check is handled by Ownable(initialOwner)

        AAVE_POOL = IAaveV3Pool(aavePool);

        for (uint256 i = 0; i < routers.length; i++) {
            if (routers[i] == address(0)) revert ZeroAddress();
            approvedRouters[routers[i]] = true;
            emit RouterApprovalSet(routers[i], true);
        }
    }

    // =========================================================================
    //                         PUBLIC ENTRY POINTS
    // =========================================================================

    /// @notice Opens a leverage position via flash loan.
    /// @dev Flash loan mode=2 (variable debt stays open). The borrowed debtAsset is
    ///      swapped to collateralAsset and supplied to Aave as collateral.
    /// @param debtAsset Token to flash borrow (e.g., USDT for LONG, WBNB for SHORT).
    /// @param flashAmount Amount to flash borrow.
    /// @param collateralAsset Token to supply as collateral after swap.
    /// @param swapRouter DEX aggregator router address (must be whitelisted).
    /// @param swapCalldata Pre-encoded swap calldata from aggregator API.
    /// @param minCollateralOut Minimum collateral tokens to receive (slippage protection).
    function openLeveragePosition(
        address debtAsset,
        uint256 flashAmount,
        address collateralAsset,
        address swapRouter,
        bytes calldata swapCalldata,
        uint256 minCollateralOut
    ) external onlyOwner whenNotPaused nonReentrant {
        if (flashAmount == 0) revert ZeroAmount();
        if (!approvedRouters[swapRouter]) revert RouterNotApproved(swapRouter);

        bytes memory params = abi.encode(OP_OPEN, swapRouter, swapCalldata, collateralAsset, minCollateralOut);

        address[] memory assets = new address[](1);
        assets[0] = debtAsset;

        uint256[] memory amounts = new uint256[](1);
        amounts[0] = flashAmount;

        uint256[] memory modes = new uint256[](1);
        modes[0] = FLASH_LOAN_MODE_VARIABLE_DEBT;

        AAVE_POOL.flashLoan(address(this), assets, amounts, modes, address(this), params, 0);
    }

    /// @notice Closes a leverage position entirely via flash loan.
    /// @dev Flash loan mode=0 (must repay in same tx). Repays debt, withdraws collateral,
    ///      swaps collateral back to debt token, repays flash loan.
    /// @param debtAsset Token to flash borrow for repaying Aave debt.
    /// @param debtAmount Amount of debt to repay (flash loan amount).
    /// @param collateralAsset Token to withdraw from Aave.
    /// @param collateralToWithdraw Amount of collateral to withdraw.
    /// @param swapRouter DEX aggregator router address (must be whitelisted).
    /// @param swapCalldata Pre-encoded swap calldata from aggregator API.
    /// @param minDebtTokenOut Minimum debt tokens received from swap (slippage protection).
    function closeLeveragePosition(
        address debtAsset,
        uint256 debtAmount,
        address collateralAsset,
        uint256 collateralToWithdraw,
        address swapRouter,
        bytes calldata swapCalldata,
        uint256 minDebtTokenOut
    ) external onlyOwner whenNotPaused nonReentrant {
        if (debtAmount == 0) revert ZeroAmount();
        if (!approvedRouters[swapRouter]) revert RouterNotApproved(swapRouter);

        bytes memory params =
            abi.encode(OP_CLOSE, swapRouter, swapCalldata, collateralAsset, collateralToWithdraw, minDebtTokenOut);

        address[] memory assets = new address[](1);
        assets[0] = debtAsset;

        uint256[] memory amounts = new uint256[](1);
        amounts[0] = debtAmount;

        uint256[] memory modes = new uint256[](1);
        modes[0] = FLASH_LOAN_MODE_NO_DEBT;

        AAVE_POOL.flashLoan(address(this), assets, amounts, modes, address(this), params, 0);
    }

    /// @notice Partially deleverages a position via flash loan.
    /// @dev Identical to closeLeveragePosition but for partial amounts.
    /// @param debtAsset Token to flash borrow for partial debt repayment.
    /// @param repayAmount Amount of debt to repay.
    /// @param collateralAsset Token to partially withdraw from Aave.
    /// @param collateralToWithdraw Amount of collateral to withdraw.
    /// @param swapRouter DEX aggregator router address (must be whitelisted).
    /// @param swapCalldata Pre-encoded swap calldata from aggregator API.
    /// @param minDebtTokenOut Minimum debt tokens received from swap (slippage protection).
    function deleveragePosition(
        address debtAsset,
        uint256 repayAmount,
        address collateralAsset,
        uint256 collateralToWithdraw,
        address swapRouter,
        bytes calldata swapCalldata,
        uint256 minDebtTokenOut
    ) external onlyOwner whenNotPaused nonReentrant {
        if (repayAmount == 0) revert ZeroAmount();
        if (!approvedRouters[swapRouter]) revert RouterNotApproved(swapRouter);

        bytes memory params =
            abi.encode(OP_DELEVERAGE, swapRouter, swapCalldata, collateralAsset, collateralToWithdraw, minDebtTokenOut);

        address[] memory assets = new address[](1);
        assets[0] = debtAsset;

        uint256[] memory amounts = new uint256[](1);
        amounts[0] = repayAmount;

        uint256[] memory modes = new uint256[](1);
        modes[0] = FLASH_LOAN_MODE_NO_DEBT;

        AAVE_POOL.flashLoan(address(this), assets, amounts, modes, address(this), params, 0);
    }

    // =========================================================================
    //                      AAVE FLASH LOAN CALLBACK
    // =========================================================================

    /// @inheritdoc IFlashLoanReceiver
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        if (msg.sender != address(AAVE_POOL)) {
            revert InvalidCaller(address(AAVE_POOL), msg.sender);
        }
        if (initiator != address(this)) {
            revert InvalidInitiator(address(this), initiator);
        }

        uint8 opType = abi.decode(params, (uint8));

        if (opType == OP_OPEN) {
            _executeOpen(assets[0], amounts[0], params);
        } else if (opType == OP_CLOSE || opType == OP_DELEVERAGE) {
            _executeCloseOrDeleverage(assets[0], amounts[0], premiums[0], opType, params);
        } else {
            revert InvalidOperationType(opType);
        }

        return true;
    }

    // =========================================================================
    //                       INTERNAL EXECUTION LOGIC
    // =========================================================================

    /// @dev Handles the OPEN flow: swap debtAsset -> collateralAsset, supply to Aave.
    ///      Mode=2: debt stays open, no repayment needed.
    function _executeOpen(address debtAsset, uint256 flashAmount, bytes calldata params) internal {
        (, // opType already decoded
            address swapRouter,
            bytes memory swapCalldata,
            address collateralAsset,
            uint256 minCollateralOut
        ) = abi.decode(params, (uint8, address, bytes, address, uint256));

        // Approve debtAsset to the swap router
        IERC20(debtAsset).forceApprove(swapRouter, flashAmount);

        // Execute swap: debtAsset -> collateralAsset
        (bool success,) = swapRouter.call(swapCalldata);
        if (!success) revert SwapFailed(swapRouter);

        // Slippage check
        uint256 collateralBalance = IERC20(collateralAsset).balanceOf(address(this));
        if (collateralBalance < minCollateralOut) {
            revert SlippageExceeded(collateralBalance, minCollateralOut);
        }

        // Supply all received collateral to Aave
        IERC20(collateralAsset).forceApprove(address(AAVE_POOL), collateralBalance);
        AAVE_POOL.supply(collateralAsset, collateralBalance, address(this), 0);

        // Revoke router approval (defense in depth)
        IERC20(debtAsset).forceApprove(swapRouter, 0);

        emit PositionOpened(debtAsset, flashAmount, collateralAsset, collateralBalance);
    }

    /// @dev Handles CLOSE and DELEVERAGE flows: repay debt, withdraw collateral,
    ///      swap collateral -> debtAsset, approve repayment to Aave.
    ///      Mode=0: Aave pulls flashAmount + premium after return.
    function _executeCloseOrDeleverage(
        address debtAsset,
        uint256 flashAmount,
        uint256 premium,
        uint8 opType,
        bytes calldata params
    ) internal {
        CloseParams memory p;
        {
            (, address sr, bytes memory sc, address ca, uint256 ctw, uint256 mdo) =
                abi.decode(params, (uint8, address, bytes, address, uint256, uint256));
            p = CloseParams({
                swapRouter: sr, swapCalldata: sc, collateralAsset: ca, collateralToWithdraw: ctw, minDebtTokenOut: mdo
            });
        }

        // Repay Aave debt with flash-borrowed tokens
        IERC20(debtAsset).forceApprove(address(AAVE_POOL), flashAmount);
        AAVE_POOL.repay(debtAsset, flashAmount, VARIABLE_RATE_MODE, address(this));

        // Withdraw collateral from Aave
        uint256 withdrawnAmount = AAVE_POOL.withdraw(p.collateralAsset, p.collateralToWithdraw, address(this));

        // Swap collateral -> debtAsset
        IERC20(p.collateralAsset).forceApprove(p.swapRouter, withdrawnAmount);
        {
            (bool success,) = p.swapRouter.call(p.swapCalldata);
            if (!success) revert SwapFailed(p.swapRouter);
        }

        // Slippage check
        {
            uint256 debtTokenBalance = IERC20(debtAsset).balanceOf(address(this));
            if (debtTokenBalance < p.minDebtTokenOut) {
                revert SlippageExceeded(debtTokenBalance, p.minDebtTokenOut);
            }
        }

        // Approve Aave Pool to pull repayment (flashAmount + premium)
        IERC20(debtAsset).forceApprove(address(AAVE_POOL), flashAmount + premium);

        // Revoke router approval
        IERC20(p.collateralAsset).forceApprove(p.swapRouter, 0);

        if (opType == OP_CLOSE) {
            emit PositionClosed(debtAsset, flashAmount, p.collateralAsset, withdrawnAmount);
        } else {
            emit PositionDeleveraged(debtAsset, flashAmount, p.collateralAsset, withdrawnAmount);
        }
    }

    // =========================================================================
    //                          ADMIN FUNCTIONS
    // =========================================================================

    /// @notice Set or revoke approval for a DEX aggregator router.
    function setRouterApproval(address router, bool approved) external onlyOwner {
        if (router == address(0)) revert ZeroAddress();
        approvedRouters[router] = approved;
        emit RouterApprovalSet(router, approved);
    }

    /// @notice Rescue tokens accidentally sent to this contract.
    function rescueTokens(address token, uint256 amount) external onlyOwner {
        if (token == address(0)) revert ZeroAddress();
        if (amount == 0) revert ZeroAmount();
        IERC20(token).safeTransfer(owner(), amount);
        emit TokensRescued(token, amount, owner());
    }

    /// @notice Pause the contract. No entry points will work while paused.
    function pause() external onlyOwner {
        _pause();
    }

    /// @notice Unpause the contract.
    function unpause() external onlyOwner {
        _unpause();
    }
}
