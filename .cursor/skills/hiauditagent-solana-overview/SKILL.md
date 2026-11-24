---
name: hiauditagent-solana-overview
description: Solana program overview and audit preparation. Triggers on hiauditagent solana overview, hiauditagent solana scope, or hiauditagent solana onboard.
---

# HiAuditAgent Solana Overview

Comprehensive Solana program onboarding and threat modeling for security engineers starting a new audit. Produces a structured overview of Anchor/native programs, account data layouts, CPI dependencies, and a prioritized audit plan tailored to Solana-specific attack surfaces.

## When to Activate

- Starting a new Solana audit
- User asks "what does this Solana program do?"
- User wants to scope/understand a Solana/Anchor codebase
- Project contains `Anchor.toml` or `Cargo.toml` with `solana-program`/`anchor-lang`

## When NOT to Use

- Solidity/EVM project (use hiauditagent-overview)
- User wants to run static analysis (use hiauditagent-analyzer)
- User wants a non-Solana application overview (use hiauditagent-overview or hiauditagent-analyzer)

## Available MCP Tools

1. **hiauditagent_scan** - Upload source and return the remote audit report
   - `path`: Directory containing the Solana program
   - The backend auto-detects Solana. Use the report plus source reading for program structure, constraints, CPI, and account layouts.

## Process

### Phase 1: Project Detection and Documentation

```
1. Check for Anchor.toml (Anchor framework) or Cargo.toml with solana-program
2. Read README.md, docs/ - program description, architecture
3. Check for previous audits in audit/ or linked reports
4. Check Cargo.lock for dependency versions (especially spl-token, anchor-lang)
5. Check test coverage - what instructions are tested?
6. Note deployment target (mainnet, devnet, localnet)
```

### Phase 2: Program Architecture Mapping

```
1. Run hiauditagent_scan, then list all programs from source
2. For each program:
   - List instruction handlers
   - Map account context structs and their constraints
   - Identify declared program IDs (declare_id!)
   - Map CPI targets (invoke/invoke_signed)
3. Map the CPI dependency graph from source after hiauditagent_scan
4. Identify program types:
   - Token program (SPL Token / Token-2022)
   - Governance program
   - Staking/Vault program
   - AMM/DEX program
   - Lending/Borrowing program
5. Note multi-program architectures and their interaction patterns
```

### Phase 3: Account Model Analysis

```
1. After hiauditagent_scan, map account data byte layouts from source
2. For each #[account] data struct:
   - Map field types and byte sizes
   - Identify authority/owner fields
   - Check for discriminator (Anchor auto-adds 8 bytes)
   - Note PDA seeds if derivable
3. For each #[derive(Accounts)] context struct:
   - Map all account fields with types and constraints
   - Identify which accounts are Signer, mutable, init, close
   - Verify has_one / constraint / seeds constraints
4. Identify accounts shared across multiple instructions
```

### Phase 4: Actor and Permission Model (Solana-specific)

| Actor | Trust Level | Permissions | Risk |
|-------|-------------|-------------|------|
| **Users** | Untrusted | Call instructions, pass accounts | Can pass wrong accounts, fake programs |
| **Authority/Admin** | Privileged | Update config, withdraw, pause | Key compromise, authority transfer |
| **PDA Signers** | Program-controlled | CPI with seeds | Seed derivation bugs, bump canonicalization |
| **External Programs** | External | CPI targets (SPL Token, etc.) | Fake program substitution, CPI return manipulation |
| **Validators** | Semi-trusted | Order transactions | Front-running, sandwich attacks |

For each privileged actor:
- What accounts do they sign?
- Can authority be transferred? Via what instruction?
- Is there a multi-sig or timelock?
- What is the maximum damage from a compromised key?

### Phase 5: Entry Point Catalog

For each instruction handler:
```
- Function name and Context struct
- Required signers
- Mutable accounts (state changes)
- CPI calls made
- Value handling (SOL transfers, token movements)
- Error conditions
```

### Phase 6: Solana-Specific Threat Model

| Threat | Description | Priority |
|--------|-------------|----------|
| **Missing signer check** | Instruction doesn't verify account is transaction signer | Critical |
| **Missing owner check** | Program doesn't verify account is owned by expected program | Critical |
| **Account substitution** | Attacker passes different account than expected | Critical |
| **Type cosplay** | Account data deserializes as wrong type (no discriminator) | High |
| **PDA seed manipulation** | Attacker influences PDA derivation seeds | High |
| **CPI to fake program** | invoke() targets attacker's program instead of SPL Token | High |
| **Uninitialized account** | Program reads data from uninitialized account | High |
| **Integer overflow** | Unchecked arithmetic in financial calculations | High |
| **Float precision** | f64/f32 rounding in token amounts | Medium |
| **Reinitialization** | Account can be initialized multiple times | Medium |
| **Account close drain** | Closed account's lamports can be drained | Medium |
| **Stale oracle data** | Pyth/Switchboard price not validated for freshness | Medium |
| **Compute exhaustion** | Nested loops or unbounded iteration | Low |
| **Hardcoded program ID** | Program ID embedded as string literal | Low |
| **Insecure randomness** | Using Clock slot/timestamp as randomness source | Low |

### Phase 7: Protocol Invariants

| Category | Invariant | Violation Impact |
|----------|-----------|-----------------|
| **Accounting** | token_vault.amount == sum(user_balances) | Fund creation/destruction |
| **Authority** | Only designated authority can call admin instructions | Unauthorized access |
| **PDA integrity** | PDAs derived with canonical bump seeds | Account collision |
| **CPI safety** | All CPI targets verified against known program IDs | Arbitrary code execution |
| **Serialization** | All deserialized accounts checked for discriminator | Type confusion |
| **Rent** | All created accounts are rent-exempt | Account deletion |

## Output Structure

```markdown
# [Program Name] - Solana Security Overview

## Summary
[2-3 sentences: what this program does, framework (Anchor/native), deployment target]

## Programs
| Program | ID | Framework | Instructions | Accounts |
|---------|----|-----------|-------------|----------|

## Architecture
- **CPI Dependencies:** SPL Token, System Program, ...
- **PDAs:** [list PDA seeds and purposes]
- **Multi-program:** [yes/no, interaction pattern]

## Account Model
### Context Structs
| Struct | Instruction | Signers | Mutable Accounts |
|--------|------------|---------|------------------|

### Data Structs
| Struct | Size | Fields | Discriminator |
|--------|------|--------|---------------|

## Actor Model
| Actor | Trust | Key Management | Max Damage |
|-------|-------|---------------|------------|

## Entry Points (Risk-Ordered)
| Instruction | Signers | CPI? | Value Flow | Risk |
|------------|---------|------|------------|------|

## Threat Model
| Threat | Likelihood | Impact | Priority |
|--------|-----------|--------|----------|

## Recommended Audit Order
1. [Highest risk instruction/module]
2. [Second highest]
...
```

## After Overview

Offer:
- "Run hiauditagent_scan for a full Solana audit?"
- "Deep dive into [highest risk instruction]?"
- "Analyze access control for all instructions?"
- "Check account data layouts for type confusion risks?"
