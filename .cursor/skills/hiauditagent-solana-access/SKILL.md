---
name: hiauditagent-solana-access
description: Solana account constraint and access control analysis. Triggers on hiauditagent solana access, hiauditagent solana permissions, hiauditagent solana constraints, or hiauditagent solana authority.
---

# HiAuditAgent Solana Access Control

Systematic analysis of Anchor account constraints, signer verification, PDA authority patterns, and account ownership checks in Solana programs. Identifies missing constraints, unverified signers, authority transfer risks, and account validation gaps.

## When to Activate

- User asks about Solana access control or permissions
- User wants to audit Anchor account constraints
- User asks "who can call this instruction?"
- User asks about authority, signer, or PDA patterns
- User wants to verify account validation is complete

## When NOT to Use

- Solidity/EVM project (use hiauditagent-access)
- User wants full analysis (use hiauditagent-analyzer)
- User wants project overview (use hiauditagent-solana-overview)
- Non-Solana application source (use hiauditagent-analyzer)

## Available MCP Tools

1. **hiauditagent_scan** - Upload source and return the remote audit report
   - `path`: Directory containing the Solana program
   - Use the report plus source reading to map constraints, signers, and instruction handlers.

## Process

### Step 1: Map the Account Constraint Model

```
1. Run hiauditagent_scan for the full project
2. Identify the constraint pattern:
   - Anchor constraints (#[account(signer)], #[account(mut, has_one = authority)])
   - Signer<'info> typed fields
   - Manual checks (account.is_signer, owner == expected)
   - PDA-based authority (seeds, bump)
3. For each context struct, verify:
   - All privileged fields have signer constraints
   - Mutable accounts have proper has_one/constraint checks
   - PDA accounts use correct seeds and bump
   - System accounts (token_program, system_program) are verified
```

### Step 2: Analyze Each Instruction Handler

For every instruction:

```
1. Run hiauditagent_scan, then list instruction handlers from source
2. For each handler, check:
   - Does it have a Context parameter?
   - Which accounts in the context are Signer?
   - Which accounts are mutable?
   - What constraints protect each account?
3. For handlers WITHOUT context (native Solana):
   - How are accounts unpacked from the accounts array?
   - Is is_signer checked for privileged accounts?
   - Is the account owner verified?
   - Is account data properly deserialized with type checks?
```

### Step 3: Signer Verification Audit

```
Anchor programs:
- Every authority/admin/payer field should be Signer<'info>
  OR have #[account(signer)] constraint
- Check: does the signer field match the authority stored in account data?
  (e.g., has_one = authority on the data account)

Native programs:
- For each AccountInfo used as a signer:
  - Is account_info.is_signer checked?
  - Is the check done BEFORE any state modifications?
  - Is the check done for EVERY code path?
```

### Step 4: Account Owner Verification

```
Critical check for native Solana programs:

For each account that holds program data:
1. Is account.owner == program_id verified?
2. Is the check done BEFORE deserialization?
3. Can an attacker pass an account owned by a different program
   with matching data layout? (Type confusion)

Anchor handles this via Account<'info, T> which auto-checks
owner + discriminator. But UncheckedAccount and AccountInfo do NOT.
```

### Step 5: PDA Authority Analysis

```
For each PDA used in the program:
1. What seeds derive this PDA?
   - Are seeds unique per user/instance?
   - Can any seed be attacker-controlled?
2. Is the canonical bump used?
   - find_program_address returns canonical bump
   - create_program_address with non-canonical bump = collision risk
3. Is the PDA used as a signer for CPI?
   - Are invoke_signed seeds correct?
   - Can the PDA be reused across programs?
4. PDA sharing risk:
   - Can two different PDAs resolve to the same address?
   - Are seeds sufficiently unique?
```

### Step 6: CPI Authority Verification

```
For each Cross-Program Invocation:
1. Is the target program ID verified?
   - token_program.key == &spl_token::id() ?
   - Or is there an Anchor Program<'info, Token> constraint?
2. Is the CPI result checked?
   - invoke()? or invoke_signed()? (with ? operator)
   - Or is the Result silently discarded?
3. For CPI with PDA signer:
   - Are the signer seeds correct?
   - Could the CPI be redirected to a different program?
```

## Anchor Constraint Checklist

### Account Types and Their Guarantees

| Type | Owner Check | Discriminator | Deserialize | Signer |
|------|------------|---------------|-------------|--------|
| `Account<'info, T>` | YES | YES (8 bytes) | YES | No |
| `Signer<'info>` | No | No | No | YES |
| `SystemAccount<'info>` | system_program | No | No | No |
| `Program<'info, T>` | BPF Loader | No | No | No |
| `AccountInfo<'info>` | **NONE** | **NONE** | **NONE** | **NONE** |
| `UncheckedAccount<'info>` | **NONE** | **NONE** | **NONE** | **NONE** |

### Critical Constraint Patterns

```rust
// Signer verification
pub authority: Signer<'info>,

// Owner match (account data references the signer)
#[account(mut, has_one = authority)]
pub vault: Account<'info, VaultData>,

// PDA derivation with seeds
#[account(
    seeds = [b"vault", user.key().as_ref()],
    bump = vault.bump,
)]
pub vault_pda: Account<'info, VaultData>,

// Init with payer and space
#[account(init, payer = payer, space = 8 + VaultData::INIT_SPACE)]
pub vault: Account<'info, VaultData>,

// Close account (return rent to destination)
#[account(mut, close = destination)]
pub account_to_close: Account<'info, SomeData>,

// Token program verification
pub token_program: Program<'info, Token>,
```

### Common Solana Access Control Vulnerabilities

| Vulnerability | Pattern | Detector |
|--------------|---------|----------|
| Missing signer on authority | `AccountInfo<'info>` for admin field | `sol-missing-signer-check` |
| Missing owner check | No `Account<T>` or manual owner verify | `sol-missing-owner-check` |
| Missing admin check | No authority verification on privileged ix | `sol-missing-admin-check` |
| Unverified token program | `AccountInfo` instead of `Program<Token>` | `sol-unverified-token-program` |
| Arbitrary CPI | invoke() without program ID verification | `sol-arbitrary-cpi` |
| Missing writable | State-changing without `#[account(mut)]` | `sol-missing-writable-check` |
| Account close drain | Close without proper lamport handling | `sol-account-close-vulnerability` |
| PDA bump non-canonical | create_program_address with stored bump | `sol-pda-bump-canonicalization` |
| Duplicate mut accounts | Same type mutable twice, no distinction | `sol-duplicate-mutable-accounts` |
| Staking validator | delegate_stake without vote account check | `sol-staking-validator-check` |
| Instruction sysvar | Accessing data without sysvar key check | `sol-missing-instruction-sysvar-check` |

## Output Format

```markdown
## Solana Access Control Analysis

### Constraint Model
- **Framework:** Anchor v0.29.0
- **Pattern:** PDA-based authority with Signer verification
- **Programs:** 2 (main program + governance)

### Account Constraints Matrix

| Context | Field | Type | Signer | Mutable | Constraints |
|---------|-------|------|--------|---------|-------------|
| Initialize | authority | Signer<'info> | YES | NO | - |
| Initialize | vault | Account<VaultData> | NO | YES | init, payer=authority |
| Withdraw | authority | Signer<'info> | YES | NO | - |
| Withdraw | vault | Account<VaultData> | NO | YES | has_one=authority, mut |

### Findings

#### [SOL-AC-01] Missing signer check on admin field - CRITICAL
**Location:** instructions/update_config.rs:15
**Issue:** `admin` field is `AccountInfo<'info>` without signer constraint
**Impact:** Anyone can call update_config with any account as admin
**Fix:** Change to `Signer<'info>` or add `#[account(signer)]`

#### [SOL-AC-02] PDA authority without has_one - HIGH
**Location:** instructions/withdraw.rs:8
**Issue:** vault account is mutable but doesn't verify `has_one = authority`
**Impact:** Attacker can withdraw from any vault by passing their own authority
**Fix:** Add `#[account(mut, has_one = authority)]` to vault field

### Summary
- **Instructions analyzed:** 6
- **With proper signer constraints:** 4
- **Without signer constraints:** 2 (CRITICAL)
- **PDA authority issues:** 1
```

## After Analysis

Offer:
- "Found constraint issues. Want me to write detailed findings?" (use hiauditagent-report)
- "Want me to validate if [instruction] is exploitable?" (use hiauditagent-validate)
- "Want me to check the full program with hiauditagent_scan?"
- "Want me to map account data layouts for type confusion risks?"
