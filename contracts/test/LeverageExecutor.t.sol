// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {LeverageExecutor} from "../src/LeverageExecutor.sol";

// =========================================================================
//                    UNIT TESTS (no fork required)
// =========================================================================

contract LeverageExecutorTest is Test {
    // BSC Mainnet addresses (used as constructor args, not called in unit tests)
    address constant AAVE_POOL = 0x6807dc923806fE8Fd134338EABCA509979a7e0cB;
    address constant WBNB = 0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c;
    address constant USDT = 0x55d398326f99059fF775485246999027B3197955;

    address constant ONEINCH_ROUTER = 0x111111125421cA6dc452d289314280a0f8842A65;
    address constant OPENOCEAN_ROUTER = 0x6352a56caadC4F1E25CD6c75970Fa768A3304e64;
    address constant PARASWAP_ROUTER = 0xDEF171Fe48CF0115B1d80b88dc8eAB59176FEe57;

    LeverageExecutor public executor;
    address public owner;
    address public attacker;

    function setUp() public {
        owner = address(this);
        attacker = makeAddr("attacker");

        address[] memory routers = new address[](3);
        routers[0] = ONEINCH_ROUTER;
        routers[1] = OPENOCEAN_ROUTER;
        routers[2] = PARASWAP_ROUTER;

        executor = new LeverageExecutor(AAVE_POOL, owner, routers);
    }

    // ----- Constructor -----

    function test_constructor_setsAavePool() public view {
        assertEq(address(executor.AAVE_POOL()), AAVE_POOL);
    }

    function test_constructor_setsOwner() public view {
        assertEq(executor.owner(), owner);
    }

    function test_constructor_approvesRouters() public view {
        assertTrue(executor.approvedRouters(ONEINCH_ROUTER));
        assertTrue(executor.approvedRouters(OPENOCEAN_ROUTER));
        assertTrue(executor.approvedRouters(PARASWAP_ROUTER));
    }

    function test_constructor_reverts_zeroPool() public {
        address[] memory routers = new address[](0);
        vm.expectRevert(LeverageExecutor.ZeroAddress.selector);
        new LeverageExecutor(address(0), owner, routers);
    }

    function test_constructor_reverts_zeroOwner() public {
        address[] memory routers = new address[](0);
        vm.expectRevert(abi.encodeWithSignature("OwnableInvalidOwner(address)", address(0)));
        new LeverageExecutor(AAVE_POOL, address(0), routers);
    }

    // ----- Access Control -----

    function test_openPosition_reverts_nonOwner() public {
        vm.prank(attacker);
        vm.expectRevert(abi.encodeWithSignature("OwnableUnauthorizedAccount(address)", attacker));
        executor.openLeveragePosition(USDT, 1000e18, WBNB, ONEINCH_ROUTER, "", 0);
    }

    function test_closePosition_reverts_nonOwner() public {
        vm.prank(attacker);
        vm.expectRevert(abi.encodeWithSignature("OwnableUnauthorizedAccount(address)", attacker));
        executor.closeLeveragePosition(USDT, 1000e18, WBNB, 1e18, ONEINCH_ROUTER, "", 0);
    }

    function test_deleveragePosition_reverts_nonOwner() public {
        vm.prank(attacker);
        vm.expectRevert(abi.encodeWithSignature("OwnableUnauthorizedAccount(address)", attacker));
        executor.deleveragePosition(USDT, 500e18, WBNB, 1e18, ONEINCH_ROUTER, "", 0);
    }

    function test_setRouterApproval_reverts_nonOwner() public {
        vm.prank(attacker);
        vm.expectRevert(abi.encodeWithSignature("OwnableUnauthorizedAccount(address)", attacker));
        executor.setRouterApproval(makeAddr("newRouter"), true);
    }

    function test_rescueTokens_reverts_nonOwner() public {
        vm.prank(attacker);
        vm.expectRevert(abi.encodeWithSignature("OwnableUnauthorizedAccount(address)", attacker));
        executor.rescueTokens(USDT, 100e18);
    }

    // ----- Router Whitelist -----

    function test_openPosition_reverts_unapprovedRouter() public {
        address fakeRouter = makeAddr("fakeRouter");
        vm.expectRevert(abi.encodeWithSelector(LeverageExecutor.RouterNotApproved.selector, fakeRouter));
        executor.openLeveragePosition(USDT, 1000e18, WBNB, fakeRouter, "", 0);
    }

    function test_setRouterApproval_works() public {
        address newRouter = makeAddr("newRouter");
        assertFalse(executor.approvedRouters(newRouter));

        executor.setRouterApproval(newRouter, true);
        assertTrue(executor.approvedRouters(newRouter));

        executor.setRouterApproval(newRouter, false);
        assertFalse(executor.approvedRouters(newRouter));
    }

    function test_setRouterApproval_reverts_zeroAddress() public {
        vm.expectRevert(LeverageExecutor.ZeroAddress.selector);
        executor.setRouterApproval(address(0), true);
    }

    // ----- Pausable -----

    function test_pause_blocksOpen() public {
        executor.pause();
        vm.expectRevert(abi.encodeWithSignature("EnforcedPause()"));
        executor.openLeveragePosition(USDT, 1000e18, WBNB, ONEINCH_ROUTER, "", 0);
    }

    function test_pause_blocksClose() public {
        executor.pause();
        vm.expectRevert(abi.encodeWithSignature("EnforcedPause()"));
        executor.closeLeveragePosition(USDT, 1000e18, WBNB, 1e18, ONEINCH_ROUTER, "", 0);
    }

    function test_pause_blocksDeleverage() public {
        executor.pause();
        vm.expectRevert(abi.encodeWithSignature("EnforcedPause()"));
        executor.deleveragePosition(USDT, 500e18, WBNB, 1e18, ONEINCH_ROUTER, "", 0);
    }

    // ----- executeOperation Caller Validation -----

    function test_executeOperation_reverts_wrongCaller() public {
        address[] memory assets = new address[](1);
        uint256[] memory amounts = new uint256[](1);
        uint256[] memory premiums = new uint256[](1);

        vm.prank(attacker);
        vm.expectRevert(abi.encodeWithSelector(LeverageExecutor.InvalidCaller.selector, AAVE_POOL, attacker));
        executor.executeOperation(assets, amounts, premiums, address(executor), "");
    }

    function test_executeOperation_reverts_wrongInitiator() public {
        address[] memory assets = new address[](1);
        uint256[] memory amounts = new uint256[](1);
        uint256[] memory premiums = new uint256[](1);

        vm.prank(AAVE_POOL);
        vm.expectRevert(abi.encodeWithSelector(LeverageExecutor.InvalidInitiator.selector, address(executor), attacker));
        executor.executeOperation(assets, amounts, premiums, attacker, "");
    }

    // ----- Zero Amount -----

    function test_openPosition_reverts_zeroAmount() public {
        vm.expectRevert(LeverageExecutor.ZeroAmount.selector);
        executor.openLeveragePosition(USDT, 0, WBNB, ONEINCH_ROUTER, "", 0);
    }

    function test_closePosition_reverts_zeroAmount() public {
        vm.expectRevert(LeverageExecutor.ZeroAmount.selector);
        executor.closeLeveragePosition(USDT, 0, WBNB, 1e18, ONEINCH_ROUTER, "", 0);
    }

    function test_deleveragePosition_reverts_zeroAmount() public {
        vm.expectRevert(LeverageExecutor.ZeroAmount.selector);
        executor.deleveragePosition(USDT, 0, WBNB, 1e18, ONEINCH_ROUTER, "", 0);
    }

    // ----- Rescue Tokens -----

    function test_rescueTokens_reverts_zeroAddress() public {
        vm.expectRevert(LeverageExecutor.ZeroAddress.selector);
        executor.rescueTokens(address(0), 100);
    }

    function test_rescueTokens_reverts_zeroAmount() public {
        vm.expectRevert(LeverageExecutor.ZeroAmount.selector);
        executor.rescueTokens(USDT, 0);
    }
}

// =========================================================================
//                   FORK TESTS (require BSC RPC)
// =========================================================================

contract LeverageExecutorForkTest is Test {
    address constant AAVE_POOL = 0x6807dc923806fE8Fd134338EABCA509979a7e0cB;
    address constant WBNB = 0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c;
    address constant USDT = 0x55d398326f99059fF775485246999027B3197955;
    address constant USDC = 0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d;

    address constant ONEINCH_ROUTER = 0x111111125421cA6dc452d289314280a0f8842A65;
    address constant OPENOCEAN_ROUTER = 0x6352a56caadC4F1E25CD6c75970Fa768A3304e64;
    address constant PARASWAP_ROUTER = 0xDEF171Fe48CF0115B1d80b88dc8eAB59176FEe57;

    LeverageExecutor public executor;
    address public owner;

    function setUp() public {
        owner = address(this);

        address[] memory routers = new address[](3);
        routers[0] = ONEINCH_ROUTER;
        routers[1] = OPENOCEAN_ROUTER;
        routers[2] = PARASWAP_ROUTER;

        executor = new LeverageExecutor(AAVE_POOL, owner, routers);
    }

    function test_fork_rescueTokens() public {
        deal(USDT, address(executor), 1000e18);
        assertEq(IERC20(USDT).balanceOf(address(executor)), 1000e18);

        uint256 ownerBalBefore = IERC20(USDT).balanceOf(owner);
        executor.rescueTokens(USDT, 1000e18);
        uint256 ownerBalAfter = IERC20(USDT).balanceOf(owner);

        assertEq(ownerBalAfter - ownerBalBefore, 1000e18);
        assertEq(IERC20(USDT).balanceOf(address(executor)), 0);
    }

    function test_fork_openLong_withMockSwap() public {
        MockSwapRouter mockRouter = new MockSwapRouter();
        executor.setRouterApproval(address(mockRouter), true);

        uint256 flashAmount = 1000e18; // 1000 USDT
        uint256 wbnbOut = 2e18; // Mock gives 2 WBNB

        // Seed the mock router with WBNB to simulate swap output
        deal(WBNB, address(mockRouter), wbnbOut);

        bytes memory swapCalldata =
            abi.encodeWithSelector(MockSwapRouter.swap.selector, USDT, WBNB, flashAmount, wbnbOut);

        executor.openLeveragePosition(USDT, flashAmount, WBNB, address(mockRouter), swapCalldata, wbnbOut);

        // Executor should not hold raw WBNB — it's all supplied to Aave
        assertEq(IERC20(WBNB).balanceOf(address(executor)), 0, "WBNB dust in executor");
    }

    function test_fork_openShort_withMockSwap() public {
        // SHORT: borrow WBNB, swap to USDC, supply USDC as collateral.
        // Use USDC (not USDT) to avoid isolation-mode restrictions on BSC.
        // Use realistic amounts: 0.5 WBNB (~$370) debt vs 500 USDC collateral.
        MockSwapRouter mockRouter = new MockSwapRouter();
        executor.setRouterApproval(address(mockRouter), true);

        uint256 flashAmount = 5e17; // 0.5 WBNB
        uint256 usdcOut = 500e18; // Mock gives 500 USDC

        deal(USDC, address(mockRouter), usdcOut);

        bytes memory swapCalldata =
            abi.encodeWithSelector(MockSwapRouter.swap.selector, WBNB, USDC, flashAmount, usdcOut);

        executor.openLeveragePosition(WBNB, flashAmount, USDC, address(mockRouter), swapCalldata, usdcOut);

        assertEq(IERC20(USDC).balanceOf(address(executor)), 0, "USDC dust in executor");
    }

    function test_fork_openPosition_slippageReverts() public {
        MockSwapRouter mockRouter = new MockSwapRouter();
        executor.setRouterApproval(address(mockRouter), true);

        uint256 flashAmount = 1000e18;
        uint256 wbnbOut = 2e18;
        uint256 minCollateralOut = 3e18; // More than router will give

        deal(WBNB, address(mockRouter), wbnbOut);

        bytes memory swapCalldata =
            abi.encodeWithSelector(MockSwapRouter.swap.selector, USDT, WBNB, flashAmount, wbnbOut);

        vm.expectRevert();
        executor.openLeveragePosition(USDT, flashAmount, WBNB, address(mockRouter), swapCalldata, minCollateralOut);
    }

    function test_fork_swapFailureReverts() public {
        FailingSwapRouter failRouter = new FailingSwapRouter();
        executor.setRouterApproval(address(failRouter), true);

        vm.expectRevert();
        executor.openLeveragePosition(
            USDT, 1000e18, WBNB, address(failRouter), abi.encodeWithSelector(FailingSwapRouter.swap.selector), 1e18
        );
    }
}

// =========================================================================
//                      MOCK CONTRACTS FOR TESTING
// =========================================================================

/// @notice Mock swap router that simulates a DEX swap by pulling tokenIn and pushing tokenOut.
contract MockSwapRouter {
    function swap(address tokenIn, address tokenOut, uint256 amountIn, uint256 amountOut) external {
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);
        IERC20(tokenOut).transfer(msg.sender, amountOut);
    }
}

/// @notice Mock router that always fails.
contract FailingSwapRouter {
    function swap() external pure {
        revert("Swap failed intentionally");
    }
}
