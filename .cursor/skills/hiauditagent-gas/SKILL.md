---
name: hiauditagent-gas
description: Gas optimization analysis and implementation for Solidity contracts. Triggers on hiauditagent gas, hiauditagent optimize, or hiauditagent efficiency.
---

# HiAuditAgent Gas Optimizer

Chain-aware gas optimization that balances real savings against code readability and security. Prioritizes impactful optimizations, filters noise by target chain economics, and never compromises security for gas savings.

## When to Activate

- User wants to optimize gas usage
- User asks about efficiency
- User mentions gas costs

## When NOT to Use

- **Solana projects** - Run `hiauditagent scan` (the backend auto-detects Anchor.toml). Gas skill focuses on EVM; for Solana compute-unit optimization, use findings from that scan.
- **Non-EVM projects** (Tron, Move, CosmWasm, TON, etc.) - Not supported. Stop and inform user.
- **No .sol files** - Nothing to analyze
- **User wants security analysis** (→ hiauditagent-analyzer)

## Workflow

1. **Detect Chain** - CRITICAL: Identify target chain BEFORE analysis
2. **Scan** - Run `hiauditagent_scan` and use the gas findings from the report
3. **Filter** - Remove findings irrelevant to target chain
4. **Security Check** - Verify no optimization introduces vulnerabilities
5. **Summarize** - Quick count by impact level (chain-adjusted)
6. **Ask** - "Fix high impact? Medium? All? Pick specific?"
7. **Fix** - Implement directly in code
8. **Verify** - Run tests after every change

## Step 1: Chain Detection (CRITICAL - DO NOT SKIP)

**Before any gas analysis, determine the target chain.**

### Detection Methods

1. **Config files** (most reliable):
   ```
   hardhat.config.js/ts → networks: { mainnet, arbitrum, optimism, polygon, bsc }
   foundry.toml → [rpc_endpoints] or comments
   truffle-config.js → networks section
   ```

2. **README.md** - Look for deployment targets, chain mentions

3. **Dependencies/Files**:
   - `foundry.toml`, `hardhat.config.*`, `truffle-config.js` → **EVM (Supported)**
   - `@arbitrum/*` → Arbitrum L2
   - `@eth-optimism/*` → Optimism L2
   - `Anchor.toml`, `*.rs` → Solana (Supported — auto-detected, uses Solana pipeline)
   - `Move.toml`, `*.move` → Move (NOT SUPPORTED)
   - `tronbox`, `tronweb` → Tron (NOT SUPPORTED)
   - No `.sol` files → NOT SUPPORTED

4. **Ask user if unclear**: "Which EVM chain is this for? (Ethereum/Arbitrum/Optimism/Polygon/BSC/Other)"
   - If user says non-EVM (Tron, Solana, etc.) → Not supported, stop.

### Chain Categories

| Category | Chains | Supported |
|----------|--------|-----------|
| **EVM L1** | Ethereum mainnet | Full |
| **EVM L2** | Arbitrum, Optimism, Base, zkSync, Scroll | Filtered |
| **EVM L1 (cheap)** | Polygon, BSC, Avalanche, Fantom | Filtered |
| **Solana** | Solana (Anchor, native) | Supported (separate pipeline) |
| **Non-EVM** | Tron, Aptos, Sui, CosmWasm, TON, etc. | NO |

**HiAuditAgent supports Solidity on EVM chains and Solana (Anchor/native) programs.**

## Chain-Specific Optimization Profiles

### EVM L1 (Ethereum Mainnet)

All standard gas optimizations apply with full impact:

| Optimization | Gas Saved | Priority |
|-------------|-----------|----------|
| Storage caching (repeated SLOAD) | 2100+ per cached read | **HIGH** |
| Struct packing (same slot) | 20000+ per packed write | **HIGH** |
| Calldata vs memory for read-only | 200-600 per param | **HIGH** |
| Unchecked math in bounded loops | ~60 per iteration | **MEDIUM** |
| Short-circuit reordering | Variable | **MEDIUM** |
| Array length caching | ~100 per loop | **MEDIUM** |
| Custom errors vs require strings | ~50 per revert | **LOW** |
| Prefix increment (++i vs i++) | ~5 per iteration | **LOW** |

### EVM L2 (Arbitrum, Optimism, Base, zkSync)

**Execution is cheap. Calldata dominates costs.**

| Optimization | Priority | Why |
|--------------|----------|-----|
| Calldata size reduction | **HIGH** | Calldata posted to L1, most expensive component |
| Short strings/bytes32 | **HIGH** | Reduces calldata bytes |
| Event data size | **HIGH** | Log data posted to L1 |
| Batch operations | **HIGH** | Reduces per-transaction L1 overhead |
| Storage caching | Low | Execution gas is cheap on L2 |
| Unchecked math | Low | Saves ~20 gas, negligible cost |
| Struct packing | Maybe | Only if reduces calldata footprint |

### EVM L1 Cheap (Polygon, BSC, Avalanche)

**Gas is cheap - micro-optimizations often cost more in readability than they save.**

| Optimization | Report? | Why |
|--------------|---------|-----|
| Storage caching | Maybe | Only for hot paths with >5 reads |
| Unchecked math | **Skip** | Saves fractions of a cent, not worth readability cost |
| Prefix increment | **Skip** | Negligible savings |
| Array length caching | **Skip** | Marginal benefit |

**Rule:** Only report if saves >1000 gas or is on a demonstrably hot path.

### Non-EVM Chains (NOT SUPPORTED)

**HiAuditAgent analyzes Solidity on EVM only.**

If detected:
```
This project targets a non-EVM chain.
HiAuditAgent supports Solidity on EVM chains only.
Gas optimization skill does not apply.
```

### Multi-chain / Unknown

1. Ask: "Which EVM chain should I optimize for?"
2. If multi-chain, optimize for the most expensive (usually Ethereum mainnet)
3. Note that L2-specific optimizations may differ from L1

## Security-Gas Tradeoff Analysis (CRITICAL)

**Never recommend an optimization that degrades security. When there is any tension between gas savings and safety, safety wins unconditionally.**

### Optimizations That Can Introduce Vulnerabilities

| Optimization | Security Risk | Rule |
|-------------|--------------|------|
| `unchecked` blocks | Integer overflow/underflow | Only in provably bounded contexts (loop counters, post-validation math) |
| Removing redundant checks | Guard removal may create exploit path | Verify the check is genuinely redundant across ALL call paths |
| Packing structs | Slot collision in upgradeable contracts | Verify no storage layout conflicts with proxy |
| Assembly (`mload`/`mstore`) | Memory corruption, reentrancy | Only recommend if gas savings are substantial (>5000) and logic is simple |
| Reducing calldata validation | Missing input sanitization | Never remove validation for gas savings |
| Short-circuiting checks | Changed revert conditions | Ensure revert behavior is equivalent |

### Gas Griefing and DoS Awareness

Flag these gas-related security issues even during optimization:

- **Unbounded loops** - Can an attacker force O(n) gas consumption?
- **Block gas limit attacks** - Can growing storage make functions exceed block limit?
- **Returnbomb attacks** - Does the code copy arbitrary-length return data?
- **Storage bloat** - Can an attacker fill mappings/arrays cheaply to grief future operations?
- **Expensive reverts** - Do failed transactions still consume significant gas?

### What to NEVER Optimize Away

```solidity
// NEVER remove reentrancy guards for gas savings
function withdraw() external nonReentrant { ... } // ~10k gas, protects against fund theft

// NEVER remove SafeERC20 wrappers
IERC20(token).safeTransfer(to, amount); // ~200 gas overhead, prevents silent failures

// NEVER remove zero-address checks on critical parameters
require(newOwner != address(0)); // ~200 gas, prevents permanent lockout

// NEVER remove CEI ordering for gas efficiency
balances[msg.sender] -= amount; // Effect FIRST
(bool ok,) = msg.sender.call{value: amount}(""); // Interaction SECOND

// NEVER use unchecked with user-controlled values
unchecked { userInput - something; } // User could cause underflow
```

## Impact Ranking Framework

Rate each optimization using:

```
Impact Score = Gas Saved × Call Frequency × Chain Multiplier

Where:
- Gas Saved = estimated gas reduction per call
- Call Frequency = how often this code path executes (high/medium/low)
- Chain Multiplier = 1.0 (L1), 0.1 (L2 execution), 5.0 (L2 calldata)
```

### Ethereum Mainnet

**High Impact** (>2000 gas saved, or >500 on hot path):
- Storage caching (repeated SLOAD)
- Struct packing
- Calldata vs memory for read-only parameters
- Unchecked math in bounded loops (only where provably safe)

**Medium Impact** (500-2000 gas saved):
- Array length caching
- Short-circuit reordering
- Custom errors replacing require strings

**Low Impact** (<500 gas saved, usually skip):
- Prefix increment (saves 5 gas)
- Minor syntax tweaks
- Single-use optimizations on cold paths

### L2s (Arbitrum, Optimism, Base)

**High Impact:**
- Calldata size reduction (bytes32 vs string, packed structs in calldata)
- Event data minimization
- Batch operations reducing L1 posting overhead

**Low Impact (often skip):**
- Execution-only optimizations (storage caching, unchecked math)
- Most "save 100 execution gas" tweaks

### Cheap L1s (Polygon, BSC)

**Only report if:**
- Saves >1000 gas per call
- On a hot path (called frequently by many users)
- Otherwise, skip - not worth the readability trade-off

## Output

**Always include chain context and security assessment:**

```
Gas analysis for: Arbitrum (L2)

Raw findings: 12
After chain filter: 4 relevant
After security review: 4 safe to apply

High (2):   calldata size reduction (2) - safe, no security impact
Medium (2): event optimization (2) - safe, no security impact
Filtered out (8): storage caching, unchecked math - low impact on L2

Security notes:
- No optimizations touch security-critical code paths
- All unchecked suggestions verified as bounded contexts

Fix high impact issues? [yes/specific/skip]
```

**For non-EVM chains:**
```
This project targets a non-EVM chain.
HiAuditAgent supports Solidity on EVM chains only.
Gas optimization skill does not apply.
```

## Implementation Rules

When user confirms fixes:
1. Fix one file at a time
2. **Before each fix:** Verify the optimization doesn't touch security-critical code
3. **After each fix:** Run full test suite
4. If tests fail, revert immediately and explain why
5. **After all fixes:** Summarize gas savings and confirm no security regressions

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "Gas optimizations are universal" | L2s have different cost models. Non-EVM is not supported. |
| "I'll assume Ethereum mainnet" | ASK or DETECT first. Wrong chain = wrong advice. |
| "Storage caching always helps" | On L2s execution is cheap; saves negligible amounts. |
| "More optimizations = better report" | Irrelevant findings waste developer time and reduce trust. |
| "Unchecked math is always good" | Only safe in provably bounded contexts. User-controlled values must be checked. |
| "Non-EVM is similar enough" | HiAuditAgent = Solidity + EVM only. Non-EVM = not supported. |
| "I can still give general advice" | If not supported, say "not supported" and stop. |
| "This reentrancy guard wastes gas" | Security > gas. Period. Never optimize away safety. |
| "The tests pass so the optimization is safe" | Tests can have gaps. Reason about correctness independent of tests. |
| "Assembly is always faster" | Assembly bypasses compiler safety checks. Only use for substantial, well-understood savings. |
