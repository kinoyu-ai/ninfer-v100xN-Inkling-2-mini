---
name: hiauditagent-access
description: Access control analysis and privilege escalation auditing for Solidity contracts. Triggers on hiauditagent access, hiauditagent permissions, hiauditagent roles, or hiauditagent privilege.
---

# HiAuditAgent Access Control

Systematic access control analysis that maps every permission boundary in a smart contract system. Identifies unprotected sensitive functions, privilege escalation paths, missing modifiers, and admin centralization risks. Aligns with the threat modeling practices used at top-tier audit firms.

## When to Activate

- User asks about access control or permissions
- User wants to audit role-based access
- User asks "who can call this function?"
- User asks about admin powers or centralization
- User wants to map the permission model
- User mentions privilege escalation

## When NOT to Use

- User wants full security analysis (→ hiauditagent-analyzer)
- User wants to understand code (→ hiauditagent-explainer)
- User wants project overview (→ hiauditagent-overview)
- User wants gas optimization (→ hiauditagent-gas)

## Available MCP Tools

1. **hiauditagent_scan** - Upload source and return the remote audit report
   - `path`: File or directory to upload (optional, defaults to current)
   - Use the report plus source reading to map permissions, modifiers, and call-chain enforcement.

## Process

### Step 1: Map the Access Control Model

```
1. Run hiauditagent_scan() for the full project
2. Identify the access control pattern:
   - Ownable (single owner)
   - Ownable2Step (two-step ownership transfer)
   - AccessControl (role-based, OpenZeppelin)
   - Custom roles (project-specific)
   - Multi-sig requirement
   - Timelock
   - Governance
3. Note key actors and their roles
```

### Step 2: Analyze Sensitive Functions

For every function flagged as sensitive AND unprotected:

```
1. Read the function implementation
2. Check for inline access control (msg.sender checks, role checks)
3. Check if access is enforced in a calling function (trace callers in source after hiauditagent_scan)
4. Determine if the function SHOULD be restricted:
   - Does it modify critical state? (balances, owners, rates)
   - Does it handle value? (ETH transfers, token minting)
   - Does it control protocol operation? (pause, upgrade, fee changes)
5. Verdict: genuinely unprotected vs. intentionally public
```

### Step 3: Review Modifier Implementations

```
For each access control modifier used in the project:
1. Read the actual implementation (not just the name)
2. Verify it correctly restricts access
3. Check for bypass vectors:
   - Can the modifier be circumvented via delegatecall?
   - Does it check the right msg.sender/tx.origin?
   - Is the role/owner state variable properly protected?
4. Check inheritance: is the modifier overridden anywhere?
```

### Step 4: Privilege Escalation Analysis

```
Map paths where a lower-privileged actor could gain higher privileges:

1. Can a regular user become an admin?
   - Is there a public setOwner/setAdmin function?
   - Can governance be manipulated via flash loans?
   - Is there a flawed role assignment function?

2. Can an admin escalate beyond their intended powers?
   - Can admin upgrade to arbitrary implementation?
   - Can admin set fees to 100%?
   - Can admin withdraw user funds?

3. Can an external contract gain unexpected privileges?
   - Callback functions that inherit caller's role
   - Delegatecall to user-controlled address
   - CREATE2 address prediction to match expected addresses
```

### Step 5: Centralization Risk Assessment

```
For each privileged role:
1. What is the key management? (EOA, multisig, timelock, governance)
2. What is the maximum damage?
   - Can they drain all funds?
   - Can they freeze the protocol?
   - Can they change critical parameters without bounds?
3. Are there protective mechanisms?
   - Timelocks on critical operations
   - Parameter bounds (max fee, min delay)
   - Two-step processes for ownership changes
   - Emergency multisig override capabilities
```

## Access Control Patterns Checklist

### Ownership

| Check | What to Verify |
|-------|---------------|
| Ownership transfer | Two-step (Ownable2Step) preferred over single-step |
| Zero-address protection | Cannot transfer ownership to address(0) |
| Renounce protection | renounceOwnership() consequences understood |
| Owner capabilities | All onlyOwner functions documented with max impact |

### Role-Based (AccessControl)

| Check | What to Verify |
|-------|---------------|
| Role hierarchy | DEFAULT_ADMIN_ROLE can grant/revoke all roles |
| Role separation | Different functions use different roles (not all admin) |
| Role assignment | Only appropriate roles can grant other roles |
| Role revocation | Can roles be revoked? By whom? |
| Initial setup | All roles properly assigned in constructor/initializer |

### Timelock

| Check | What to Verify |
|-------|---------------|
| Delay enforcement | Minimum delay cannot be set to 0 |
| Cancellation | Can pending operations be cancelled? By whom? |
| Execution window | Is there a maximum execution window after delay? |
| Bypass vectors | Can any operation skip the timelock? |

### Multi-sig

| Check | What to Verify |
|-------|---------------|
| Threshold | Is the threshold appropriate (e.g., 3-of-5)? |
| Owner management | Can signers be changed? What's the process? |
| Single point of failure | Can one signer block all operations? |

## Common Vulnerabilities

### Missing Access Control

```solidity
// VULNERABLE: Anyone can call
function setFee(uint256 newFee) external {
    fee = newFee;
}

// FIXED: Only owner
function setFee(uint256 newFee) external onlyOwner {
    require(newFee <= MAX_FEE, "Fee too high");
    fee = newFee;
}
```

### Unprotected Initializer

```solidity
// VULNERABLE: Can be re-initialized
function initialize(address _owner) external {
    owner = _owner;
}

// FIXED: Initializer modifier + disable in constructor
constructor() {
    _disableInitializers();
}

function initialize(address _owner) external initializer {
    __Ownable_init(_owner);
}
```

### tx.origin Authentication

```solidity
// VULNERABLE: Phishing via malicious contract
require(tx.origin == owner, "Not owner");

// FIXED: Use msg.sender
require(msg.sender == owner, "Not owner");
```

### Unsafe Delegatecall

```solidity
// VULNERABLE: Delegatecall to user-controlled address
function execute(address target, bytes calldata data) external onlyOwner {
    target.delegatecall(data); // Can change storage arbitrarily
}

// SAFER: Whitelist of allowed targets
function execute(address target, bytes calldata data) external onlyOwner {
    require(allowedTargets[target], "Not allowed");
    target.delegatecall(data);
}
```

## Output Format

```markdown
## Access Control Analysis

### Access Control Model
- **Pattern:** Ownable2Step + AccessControl (OpenZeppelin)
- **Key Management:** Owner = 3-of-5 multisig, Admin role = 2-of-3 multisig
- **Timelock:** 48-hour delay on all owner operations

### Role Map

| Role | Holder | Capabilities | Max Damage |
|------|--------|-------------|------------|
| Owner | 3-of-5 multisig | upgrade, pause, set fees | Full protocol control |
| Admin | 2-of-3 multisig | setOracle, addToken | Price manipulation |
| Keeper | EOA (bot) | liquidate, rebalance | Selective liquidation |
| User | Any address | deposit, withdraw, swap | Self-harm only |

### Findings

#### [AC-01] Missing access control on setRewardRate() - HIGH
**Location:** Staking.sol:78
**Issue:** `setRewardRate()` is external with no modifier. Any caller can set arbitrary reward rates.
**Impact:** Attacker sets reward rate to max, drains reward pool.
**Fix:** Add `onlyOwner` modifier.

#### [AC-02] Single-step ownership transfer - LOW
**Location:** Vault.sol:12 (inherits Ownable, not Ownable2Step)
**Issue:** Ownership can be transferred to wrong address in a single transaction.
**Impact:** Permanent loss of admin control if wrong address is used.
**Fix:** Inherit Ownable2Step instead of Ownable.

### Summary
- **Total public/external functions:** 24
- **State-changing without modifiers:** 3
- **Sensitive unprotected:** 1 (CRITICAL)
- **Centralization concerns:** Admin can pause + set unbounded fees (no max cap)
```

## After Analysis

Offer:
- "Found unprotected sensitive functions. Want me to write a detailed report?" (→ hiauditagent-report)
- "Want me to validate if [specific function] is exploitable?" (→ hiauditagent-validate)
- "Want me to write a PoC for the missing access control?" (→ hiauditagent-poc)

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "The modifier name says onlyOwner, must be correct" | Read the implementation. Custom modifiers can have bugs. |
| "This function isn't sensitive, no modifier needed" | Verify WHAT the function does. State-changing functions may have hidden impact. |
| "Admin is trusted, centralization isn't a finding" | Admin key compromise is real. Document all admin capabilities and their maximum damage. |
| "Access control is inherited, it's probably fine" | Verify inheritance. Overridden functions may lose modifiers. Check the full chain. |
| "tx.origin is fine for simple contracts" | tx.origin is NEVER safe for authentication. Always flag it. |
| "The function is internal, so it's protected" | Verify no public/external function calls it without access control. Trace the call graph. |
| "Initializers are only called once" | Without the initializer modifier, they can be called again. Always verify the protection. |
| "The role hierarchy looks standard" | Standard patterns can have non-standard implementations. Check role admin relationships. |
| "I only need to check admin functions" | Any state-changing function without access control is a potential vulnerability. Check ALL. |
