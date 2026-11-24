---
name: hiauditagent-simplify
description: Solidity code simplification and refactoring for clarity and maintainability. Triggers on hiauditagent simplify, hiauditagent refactor, or hiauditagent clean up.
---

# HiAuditAgent Simplify

Security-preserving code simplification that enhances clarity without compromising safety. Every refactoring maintains exact behavioral equivalence, preserves all security patterns, and is verified through the existing test suite.

## When to Activate

- Developer wants to clean up code
- User asks to simplify or refactor
- User wants to improve readability
- After writing new code that could be cleaner

## When NOT to Use

- **Unfamiliar codebase** - Understand first (→ hiauditagent-overview)
- **Security-critical changes** - If unsure about a pattern's purpose, don't touch it
- **No tests available** (Developer Mode only) - Cannot verify behavior without tests
- **Active audit** - Never modify code under active audit without explicit permission

## Two Modes

### Developer Mode (Default)
User wants to **modify** their codebase for maintainability.
- Edit actual source files
- Run tests after changes
- Commit simplified code

### Auditor Mode
User wants to **understand** complex code during audit.
- Create scratch/mental simplification for analysis
- **NEVER modify files being audited**
- Output simplified version to terminal or scratch file
- Original code stays untouched

**Detection:** If in audit context (hiauditagent-analyzer was used, or user mentions "audit"), default to Auditor Mode.

## Core Principles

### 1. Behavioral Equivalence (Non-Negotiable)

Every simplification must produce identical behavior:
- Same outputs for all inputs
- Same state changes in same order
- Same revert conditions and error messages
- Same event emissions
- Same gas profile (within reasonable tolerance)

### 2. Security Preservation (Non-Negotiable)

**NEVER simplify in a way that could weaken security. When in doubt, don't touch it.**

Protected patterns that must NEVER be simplified away:
```solidity
// KEEP: Reentrancy guards
function withdraw() external nonReentrant { ... }

// KEEP: CEI ordering (Checks-Effects-Interactions)
balances[msg.sender] -= amount;           // Effect BEFORE
(bool ok,) = msg.sender.call{value: amount}("");  // Interaction AFTER
require(ok);

// KEEP: Access control modifiers (even if they seem redundant)
function setFee(uint256 fee) external onlyOwner { ... }

// KEEP: Explicit type casts (document intent and prevent silent truncation)
uint128 small = uint128(bigNumber);

// KEEP: Unchecked blocks (gas optimization, intentional)
unchecked { ++i; }

// KEEP: SafeERC20 wrappers (handle non-compliant tokens)
token.safeTransfer(to, amount);

// KEEP: Zero-address checks (prevent permanent lockout)
require(newOwner != address(0), "Zero address");

// KEEP: Pausable checks (emergency stop mechanism)
require(!paused, "Paused");

// KEEP: Deadline/expiration checks
require(block.timestamp <= deadline, "Expired");
```

### 3. Clarity Over Brevity

Readable code is more secure than compact code:
- Avoid nested ternaries
- Use meaningful variable names
- Keep functions focused on a single responsibility
- Prefer explicit over implicit behavior

### 4. Gas Awareness

Don't "simplify" into more expensive code. Preserve:
- Storage caching patterns (avoid re-reading from storage)
- Unchecked blocks where mathematically safe
- Calldata over memory for read-only parameters
- Struct packing and storage layout

## Workflow

### Developer Mode

```
1. UNDERSTAND - Read the code. Understand WHY it's written this way.
   - Check git history for context on complex patterns
   - Read comments and documentation
   - Identify security-critical sections

2. SCOPE - Define what to simplify (specific function, file, or area)
   - Explicitly mark security-critical code as "DO NOT TOUCH"
   - List transformations to apply

3. VERIFY TESTS EXIST - If no tests cover the code, warn user
   - "No test coverage for withdraw(). Simplification risks introducing bugs.
     Write tests first, or proceed with extra caution?"

4. ANALYZE - Identify simplification opportunities
   - Map which patterns are security-relevant vs. pure complexity
   - Classify each opportunity: safe / needs-review / do-not-touch

5. SIMPLIFY - Apply transformations via Edit tool
   - One logical change at a time
   - Preserve all security patterns

6. TEST - Run full test suite after EVERY change
   - If tests fail, REVERT immediately
   - If tests pass, verify the change is semantically correct (tests may have gaps)

7. REPORT - Summarize what changed and what was preserved
```

### Auditor Mode

```
1. UNDERSTAND - Read the code and surrounding context
2. SCOPE - What to simplify for analysis?
3. ANALYZE - Identify complexity that obscures logic
4. OUTPUT - Show simplified version in terminal or scratch file
5. DO NOT EDIT ORIGINAL FILES
6. NOTE - "This is a simplified view for analysis. Original code unchanged."
7. FLAG - Note any security concerns discovered during simplification
```

## Simplification Patterns

### Pattern 1: Flatten Nesting with Early Returns

```solidity
// Before: Deep nesting hides the happy path
function withdraw(uint256 amount) external {
    if (amount > 0) {
        if (balances[msg.sender] >= amount) {
            if (!paused) {
                balances[msg.sender] -= amount;
                payable(msg.sender).transfer(amount);
            }
        }
    }
}

// After: Guards at top, happy path clear
function withdraw(uint256 amount) external {
    if (amount == 0) revert ZeroAmount();
    if (balances[msg.sender] < amount) revert InsufficientBalance();
    if (paused) revert ContractPaused();

    balances[msg.sender] -= amount;
    payable(msg.sender).transfer(amount);
}
```

**Security note:** Revert conditions must be exactly equivalent. Early returns with `revert` vs. silent no-op from nested `if` changes behavior.

### Pattern 2: Extract Repeated Logic into Modifiers

```solidity
// Before: Same checks duplicated across functions
function deposit() external payable {
    require(msg.value > 0, "Zero");
    require(!paused, "Paused");
    // ...
}
function withdraw(uint256 amt) external {
    require(amt > 0, "Zero");
    require(!paused, "Paused");
    // ...
}

// After: DRY with modifiers
modifier whenNotPaused() {
    if (paused) revert ContractPaused();
    _;
}
modifier nonZero(uint256 value) {
    if (value == 0) revert ZeroValue();
    _;
}

function deposit() external payable whenNotPaused nonZero(msg.value) { ... }
function withdraw(uint256 amt) external whenNotPaused nonZero(amt) { ... }
```

**Security note:** Modifier ordering matters. Ensure the extracted checks execute in the same order as the originals.

### Pattern 3: Name Complex Conditions

```solidity
// Before: Hard to audit
if (amount > 0 && amount <= max && !blocked[msg.sender] && block.timestamp >= start) { ... }

// After: Self-documenting, easier to verify each condition
bool isValidAmount = amount > 0 && amount <= max;
bool isAllowedUser = !blocked[msg.sender];
bool hasStarted = block.timestamp >= start;

if (isValidAmount && isAllowedUser && hasStarted) { ... }
```

### Pattern 4: Replace Magic Numbers with Named Constants

```solidity
// Before: What does 10000 mean? What does 86400 mean?
uint256 fee = amount * 300 / 10000;
require(block.timestamp > lastClaim + 86400);

// After: Intent is clear, easier to verify correctness
uint256 constant FEE_BASIS_POINTS = 300;   // 3%
uint256 constant BASIS_POINTS = 10_000;
uint256 constant CLAIM_COOLDOWN = 1 days;

uint256 fee = amount * FEE_BASIS_POINTS / BASIS_POINTS;
require(block.timestamp > lastClaim + CLAIM_COOLDOWN);
```

### Pattern 5: Decompose Long Functions

```solidity
// Before: 80-line function mixing validation, calculation, and state updates

// After: Separated concerns
function withdraw(uint256 amount) external nonReentrant {
    _validateWithdrawal(msg.sender, amount);    // checks
    uint256 shares = _calculateShares(amount);   // pure math
    _executeWithdrawal(msg.sender, shares, amount); // effects + interactions
}
```

**Security note:** Ensure decomposed functions maintain CEI ordering across the call chain. Don't accidentally put an interaction before an effect by splitting into sub-functions.

### Other Safe Simplifications
- Remove dead/commented code (verify it's truly unreachable first)
- Replace require strings with custom errors (saves gas + cleaner)
- Remove redundant initializations (`uint256 x = 0` → `uint256 x`)
- Improve variable/function names for clarity
- Add NatSpec documentation for public functions
- Consolidate duplicate struct definitions

## What NOT to Simplify

**Security patterns - preserve even if they look verbose or redundant:**

| Pattern | Why It Must Stay |
|---------|-----------------|
| Reentrancy guards (nonReentrant) | Prevents fund theft via reentrant calls |
| CEI ordering | Prevents reentrancy exploits |
| SafeERC20 wrappers | Handles non-compliant tokens (USDT, etc.) |
| Access control modifiers | Prevents unauthorized operations |
| Zero-address checks | Prevents permanent lockout |
| Explicit type casts | Documents intent, prevents silent truncation |
| Unchecked blocks | Intentional gas optimization, removing changes behavior |
| Pausable checks | Emergency stop mechanism |
| Deadline enforcement | Prevents stale transaction execution |
| Slippage protection | Prevents sandwich attacks |
| Input bounds validation | Prevents edge case exploits |

## Output

### Developer Mode Output
```
Simplified: Vault.sol

Changes applied:
1. withdraw(): 3 nested ifs → early reverts (L45-L60)
2. Created modifier: whenNotPaused (applied to 4 functions)
3. Replaced 8 require strings with custom errors (gas: -400/call)
4. Named 3 complex conditions for readability
5. Removed 12 lines confirmed dead code (git blame: unreachable since commit abc123)

Security patterns preserved:
- nonReentrant on withdraw(), deposit(), claim()
- CEI ordering in all value-transfer functions
- SafeERC20 for all token operations
- onlyOwner on all admin functions

Tests: All 47 tests passing
Next: Review changes, then commit
```

### Auditor Mode Output
```
## Simplified View: withdraw()

*For analysis only - original code unchanged*

Core logic (after stripping guards):
1. Validate: amount > 0, sufficient balance, not paused
2. Calculate: shares = amount * totalShares / totalAssets
3. Update state: balances[user] -= amount, totalShares -= shares
4. Transfer: send ETH to user

Security observations from simplification:
- CEI VIOLATION: balance updated (L48) AFTER transfer (L45)
- No reentrancy guard on this function
- Rounding favors user (rounds UP on withdrawal shares)

Potential finding: Reentrancy + rounding exploitation in withdraw()
```

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "This guard looks redundant, I'll remove it" | Guards exist for reasons. Understand the threat it mitigates before removing. |
| "I'll simplify first, understand later" | NEVER simplify code you don't fully understand. Understanding comes first. |
| "The tests pass, so my changes are safe" | Tests can have gaps. Reason about behavioral equivalence independently. |
| "This is obviously dead code" | Verify it's actually unreachable. Check all call paths and inheritance. |
| "Fewer lines = better code" | Clarity > brevity. Don't compress for line count. |
| "I'll combine these checks for efficiency" | Combined checks can mask individual failure modes. Keep them separate. |
| "This explicit cast is unnecessary" | Explicit casts document intent and prevent future bugs from type changes. |
| "The modifier is just boilerplate" | Modifiers are security boundaries. "Boilerplate" access control prevents exploits. |
| "I'll clean up the error handling" | Error paths are security-critical. Understand each revert condition before changing. |
| "This validation is overly cautious" | Defense-in-depth is intentional. "Overly cautious" is exactly the point. |
