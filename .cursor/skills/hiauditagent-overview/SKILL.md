---
name: hiauditagent-overview
description: Project overview and audit preparation for smart contract security. Triggers on hiauditagent overview, hiauditagent scope, or hiauditagent onboard.
---

# HiAuditAgent Overview

Comprehensive project onboarding and threat modeling for security engineers starting a new audit. Produces a structured threat model, attack surface map, and prioritized audit plan that mirrors the preparation process used at top security firms.

## When to Activate

- Starting a new audit
- User asks "what does this project do?"
- User wants to scope/understand a codebase

## When NOT to Use

- User wants to run static analysis (→ hiauditagent-analyzer)
- User wants to explain specific code (→ hiauditagent-explainer)
- User already understands the project and wants to audit

## Process

### Phase 1: Documentation and Prior Art Review

```
1. README.md - project description, architecture, deployment targets
2. docs/ folder if exists - specifications, design documents
3. Previous audits - check audit/, audits/, or linked reports
4. Known issues - known-issues.md, issue tracker links, inline @audit-known
5. Deployment config - foundry.toml, hardhat.config for chain targets
6. Dependencies - package.json, lib/ for external library versions
7. Test coverage - what's tested, what's NOT tested
```

**Critical:** Note any discrepancies between documentation and actual code. Documentation says X but code does Y = potential finding.

### Phase 2: Architecture Mapping

```
1. List ALL .sol files with their purpose
2. Classify: core contracts vs libraries vs interfaces vs test helpers
3. Map inheritance hierarchy (full chain, including OpenZeppelin bases)
4. Identify external dependencies with exact versions
5. Map inter-contract communication (which contracts call which)
6. Identify proxy patterns and upgradeability
7. Note deployment order dependencies
```

### Phase 3: Actor and Permission Model

**Identify every actor in the system and their permissions.**

| Actor | Trust Level | Permissions | Risk |
|-------|-------------|-------------|------|
| **Users** | Untrusted | deposit, withdraw, swap | Can be malicious, MEV-aware |
| **Admin/Owner** | Privileged | pause, setFee, upgrade | Key compromise, rug risk |
| **Governance** | Semi-trusted | propose, vote, execute | Governance attacks, flash loan voting |
| **Keepers** | Semi-trusted | liquidate, rebalance | Liveness dependency, manipulation |
| **Oracles** | External | provide price data | Manipulation, staleness, downtime |
| **External Protocols** | External | callbacks, token transfers | Upgrades, exploits, composability |

For each privileged actor:
- What is the key management? (EOA, multisig, timelock, governance)
- What damage can they do? (pause, drain, upgrade to malicious)
- Is there a timelock on critical operations?
- Can privileges be revoked or transferred?

### Phase 4: Entry Point Catalog

**Map every external/public function accessible to untrusted callers.**

#### User Functions (Highest Risk - Untrusted Input)
```
For each: function name, contract, parameters, value handling, access control
- deposit(), withdraw(), swap(), transfer()
- Any function handling ETH or token value
- Any function with user-controlled parameters affecting state
```

#### Admin Functions (Check Access Control)
```
For each: function name, contract, permission, impact if compromised
- setFee(), pause(), upgrade(), setOracle()
- What's the maximum damage from each admin function?
- Is there a timelock? Are parameters bounded?
```

#### Callbacks (Reentrancy Risk)
```
For each: function name, who calls it, what state it can access
- onFlashLoan(), uniswapV3Callback(), tokensReceived()
- Can the callback re-enter other protocol functions?
- What state is accessible during the callback?
```

#### Fallback/Receive Functions
```
- Does the contract accept ETH? Via which mechanism?
- Can unexpected ETH receipt break accounting?
```

### Phase 5: Value Flow Mapping

```
INBOUND (how value enters):
- User deposits (ETH, ERC20, ERC721)
- Protocol fees collected
- Yield from external protocols
- Flash loan proceeds

INTERNAL (how value moves):
- Between protocol contracts
- Accounting updates (shares, balances, debt)
- Fee distribution
- Reward accumulation

OUTBOUND (how value exits):
- User withdrawals
- Liquidation payouts
- Fee distributions to stakeholders
- Emergency withdrawals

EDGE CASES:
- What happens if tokens are sent directly (not via deposit)?
- Fee-on-transfer tokens: does accounting handle actual vs. expected amounts?
- Rebasing tokens: does share accounting stay consistent?
- ERC777 tokens: callback on transfer, reentrancy vector?
```

### Phase 6: Trust Boundary Analysis

```
┌─────────────────────────────────────────────────┐
│ UNTRUSTED (assume adversarial)                  │
│ - All external users and their contracts        │
│ - MEV searchers and sandwich bots               │
│ - Flash loan borrowers                          │
│ - Token contracts (may be malicious ERC20s)     │
├─────────────────────────────────────────────────┤
│ EXTERNAL DEPENDENCIES (trust but verify)        │
│ - Oracles (Chainlink, Uniswap TWAP)           │
│ - External protocols (Aave, Compound, Uniswap) │
│ - Token standards (ERC20 compliance varies)     │
│ - L2 sequencer (availability, ordering)         │
├─────────────────────────────────────────────────┤
│ PRIVILEGED (trusted but scope damage)           │
│ - Admin/Owner (key compromise scenario)         │
│ - Governance (flash loan voting scenario)       │
│ - Keepers/Bots (liveness and manipulation)      │
├─────────────────────────────────────────────────┤
│ INTERNAL (trusted)                              │
│ - Protocol's own contracts                      │
│ - Verified libraries (pinned versions)          │
└─────────────────────────────────────────────────┘
```

### Phase 7: Protocol Invariant Definition

**Define what MUST always be true for the protocol to be correct.**

| Category | Invariant | Violation Impact |
|----------|-----------|-----------------|
| **Accounting** | totalSupply == sum(balances) | Fund creation/destruction |
| **Accounting** | totalAssets >= totalLiabilities | Protocol insolvency |
| **Authorization** | Only admin can call privileged functions | Unauthorized access |
| **Economic** | shares * pricePerShare >= deposited value | Value extraction |
| **Temporal** | Timelock delay enforced on all critical ops | Governance attack |
| **Conservation** | tokens_in == tokens_out + fees | Value leak/creation |
| **Ordering** | State transitions follow defined sequence | State corruption |

### Phase 8: Risk-Based Audit Focus Areas

| Protocol Type | Critical Focus Areas |
|--------------|---------------------|
| **DeFi/Lending** | Liquidation math, interest accrual precision, oracle freshness, flash loan interactions, bad debt handling, collateral factor manipulation |
| **DEX/AMM** | Price calculation accuracy, slippage bounds, LP share math, fee extraction, sandwich resistance, concentrated liquidity edge cases |
| **Staking/Vaults** | Deposit/withdraw share accounting, first depositor attack, reward distribution precision, donation attacks, compounding errors |
| **NFT/Gaming** | Minting access control, randomness source, marketplace fee handling, royalty enforcement, metadata immutability |
| **Governance** | Voting power calculation, proposal execution safety, flash loan voting, timelock bypass, quorum manipulation |
| **Bridges** | Message verification, replay protection, censorship resistance, liquidity management, sequencer trust |
| **Upgradeable** | Storage layout preservation, initialization protection, function selector clashes, admin key security |

## Output Structure

```markdown
# [Project Name] - Security Overview

## Summary
[2-3 sentences: what this project does, its core value proposition, and deployment target]

## Architecture
| Contract | Purpose | Lines | Complexity |
|----------|---------|-------|------------|
| Vault.sol | Core deposit/withdraw logic | 350 | High |
| Router.sol | Entry point, input routing | 120 | Medium |
| Oracle.sol | Price feed integration | 80 | Medium |
| Token.sol | Protocol token (ERC20) | 60 | Low |

**Inheritance:**
- Vault → ERC4626 → ERC20 → Ownable2Step → ReentrancyGuard → Pausable
**External Deps:** OpenZeppelin v4.9.3, Chainlink v0.8

## Actor Model
| Actor | Trust | Key Management | Max Damage |
|-------|-------|---------------|------------|
| Users | None | EOA/Contract | Self-harm only |
| Admin | High | 3-of-5 multisig | Pause protocol |
| Oracle | External | Chainlink | Stale prices |

## Entry Points (Risk-Ordered)
### User Functions (Priority: CRITICAL)
| Function | Contract | Handles Value | Risk |
|----------|----------|--------------|------|
| deposit() | Vault.sol:45 | ETH inbound | Share accounting |
| withdraw() | Vault.sol:89 | ETH outbound | Reentrancy, rounding |

### Admin Functions (Priority: HIGH)
| Function | Contract | Permission | Max Impact |
|----------|----------|------------|------------|
| setFee() | Vault.sol:120 | onlyOwner | Fee up to 10% (bounded) |
| pause() | Vault.sol:130 | onlyOwner | Freeze all operations |

### Callbacks (Priority: HIGH)
- onFlashLoan() - called by flash lender, reentrancy risk

## Value Flow
- **In:** ETH via deposit(), ERC20 via depositToken()
- **Out:** ETH via withdraw(), rewards via claim()
- **Edge:** Direct ETH transfer → not counted in accounting (donation attack?)

## Trust Boundaries
- **Admin:** 3-of-5 multisig, can pause + set fees (bounded), NO timelock
- **Oracle:** Chainlink ETH/USD, 1hr heartbeat, no fallback
- **External:** Uniswap V3 for swaps, no slippage protection observed

## Protocol Invariants
1. totalShares * pricePerShare == totalAssets (accounting conservation)
2. Only owner can call admin functions (access control)
3. Withdrawals cannot exceed user's share of assets (fund safety)
4. Fee percentage bounded to [0, MAX_FEE] (economic bound)

## Threat Model Summary
| Threat | Likelihood | Impact | Priority |
|--------|-----------|--------|----------|
| Reentrancy in withdraw() | Medium | Critical | 1 |
| Oracle price manipulation | Low | High | 2 |
| First depositor share inflation | Medium | Medium | 3 |
| Admin key compromise | Low | Critical | 4 |
| Flash loan + donation attack | Medium | High | 5 |

## Recommended Audit Order
1. Vault.sol - core logic, highest value at risk, most attack surface
2. Router.sol - entry point, input validation, routing logic
3. Oracle.sol - external dependency, price feed trust
4. Token.sol - standard ERC20, lower priority unless modified

## Prior Audit Notes
- [Previous audit firm] - [date] - [X findings, Y resolved]
- Known issues: [list from known-issues.md]
```

## After Overview

Offer:
- "Run HiAuditAgent static analysis?"
- "Deep dive into [highest risk contract]?"
- "Start with invariant verification for [critical invariant]?"
- "Map call graph for [critical function]?"

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "I'll skip the libraries, they're standard" | Libraries can be modified, forked, or misused. Note version and any modifications. |
| "This looks like a simple project" | Simple projects can have complex bugs. Complexity hides in interactions, not line count. |
| "README explains everything" | README can be outdated. Verify every claim against actual code. |
| "I'll focus only on the main contract" | Entry points can be anywhere. Callbacks, fallbacks, and helper contracts all matter. |
| "Previous audit means it's safe" | Code changes since last audit. New features, new bugs. Previous audit may have missed things. |
| "I understand the pattern, no need to map it" | Every implementation differs. Map THIS codebase, not a generic version. |
| "The admin is trusted, no need to scope their powers" | Admin key compromise is a real threat. Always scope and document admin capabilities. |
| "I'll define invariants later during the audit" | Invariants guide the audit. Define them upfront to know what to verify. |
