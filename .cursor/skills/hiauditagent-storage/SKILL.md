---
name: hiauditagent-storage
description: Storage layout analysis and upgrade safety verification for Solidity contracts. Triggers on hiauditagent storage, hiauditagent layout, hiauditagent slots, or hiauditagent upgrade safety.
---

# HiAuditAgent Storage

Deep storage layout analysis for Solidity smart contracts. Maps slot assignments, detects packing opportunities, identifies storage collisions in proxy/upgradeable patterns, and verifies upgrade safety. Essential for any audit involving proxies, diamonds, or upgradeable contracts.

## When to Activate

- User asks about storage layout or slot assignments
- User is auditing an upgradeable/proxy contract
- User asks about storage collisions or packing
- User wants to verify upgrade safety between implementations
- User asks "what slot is this variable in?"

## When NOT to Use

- User wants general security analysis (→ hiauditagent-analyzer)
- User wants gas optimization (→ hiauditagent-gas) — though struct packing overlaps
- User wants contract overview (→ hiauditagent-overview)
- No upgradeable/proxy pattern present AND user didn't ask about storage

## Process

1. **Identify** - Determine if the project uses upgradeable patterns (proxy, UUPS, diamond, beacon)
2. **Map** - Run `hiauditagent_scan`, then map the slot-by-slot layout from source
3. **Analyze** - Check for collisions, packing opportunities, and upgrade risks
4. **Verify** - Compare layouts between implementation versions if applicable
5. **Report** - Structured findings with slot numbers and concrete evidence

## Available MCP Tools

1. **hiauditagent_scan** - Upload source and return the remote audit report
   - `path`: File or directory to upload (optional, defaults to current)
   - Use the report plus source reading to map storage slots, inheritance, and upgrade safety.

## Workflow

### Single Contract Analysis

```
1. Run hiauditagent_scan, then map storage slots for ContractName from source
2. Review slot assignments:
   - Are variables packed efficiently?
   - Are mappings and dynamic arrays positioned safely?
   - Do inherited variables come first in slot order?
3. Cross-reference modifiers and function context in source
4. Report findings
```

### Upgrade Safety Analysis

```
1. Identify the proxy pattern:
   - TransparentProxy: check implementation slot (EIP-1967)
   - UUPS: check _authorizeUpgrade implementation
   - Diamond: check facet storage patterns
   - Beacon: check beacon address storage

2. Map storage for BOTH old and new implementations from source after hiauditagent_scan:
   - OldImpl
   - NewImpl

3. Compare slot-by-slot:
   - NO existing variable can change type
   - NO existing variable can change position
   - New variables MUST append at the end
   - Storage gaps must be decremented correctly

4. Check for collision risks:
   - Proxy admin slot vs implementation slots
   - Diamond storage vs facet storage
   - Unstructured storage (bytes32 hash-based) conflicts
```

## Storage Layout Fundamentals

### Slot Assignment Rules (Solidity)

| Type | Slot Behavior |
|------|--------------|
| `uint256`, `int256`, `bytes32`, `address` | 1 full slot (32 bytes) |
| `uint128`, `uint64`, `uint32`, etc. | Can pack with adjacent smaller types |
| `bool` | 1 byte, packs with adjacent types |
| `mapping(K => V)` | 1 slot (for the mapping itself); values at `keccak256(key . slot)` |
| `T[]` dynamic array | 1 slot (length); elements at `keccak256(slot) + index` |
| `T[N]` fixed array | N consecutive slots (or packed) |
| `struct` | Members laid out sequentially, can pack within 32-byte boundaries |
| `string`, `bytes` | 1 slot if ≤31 bytes; otherwise like dynamic array |
| `constant` | **No slot** - compiled into bytecode |
| `immutable` | **No slot** - stored in deployed bytecode |

### Packing Rules

Variables pack into a single 32-byte slot when:
1. They are declared consecutively in the contract
2. Their combined size ≤ 32 bytes
3. Each variable fits within the remaining space in the current slot

```solidity
// Efficient: 1 slot (20 + 1 + 8 = 29 bytes, fits in 32)
address owner;    // 20 bytes
bool paused;      // 1 byte
uint64 timestamp; // 8 bytes

// Inefficient: 3 slots (bool alone wastes 31 bytes)
bool paused;       // slot 0 (wastes 31 bytes)
uint256 balance;   // slot 1 (full slot, can't pack with bool before it)
bool active;       // slot 2 (wastes 31 bytes)
```

## Security Checks

### Proxy Storage Collisions

```
EIP-1967 standard slots:
- Implementation: bytes32(uint256(keccak256('eip1967.proxy.implementation')) - 1)
- Admin:          bytes32(uint256(keccak256('eip1967.proxy.admin')) - 1)
- Beacon:         bytes32(uint256(keccak256('eip1967.proxy.beacon')) - 1)

Check:
1. Does the implementation's normal storage overlap with these special slots?
   (Extremely unlikely but verify for non-standard proxies)
2. For custom proxies: are storage slot constants unique and non-colliding?
```

### Storage Gaps for Inheritance

```solidity
// Base contract with storage gap
contract BaseV1 {
    uint256 public value;       // slot 0
    uint256[49] private __gap;  // slots 1-49 (reserved)
}

// Derived contract
contract DerivedV1 is BaseV1 {
    uint256 public extra;  // slot 50
}

// Upgrade: BaseV2 adds a variable
contract BaseV2 {
    uint256 public value;       // slot 0
    uint256 public newValue;    // slot 1 (uses gap space)
    uint256[48] private __gap;  // slots 2-49 (gap shrinks by 1)
}
```

**Verification checklist:**
- [ ] Gap size decreased by exactly the number of new variables
- [ ] New variables are added AFTER existing variables, BEFORE the gap
- [ ] Gap naming is consistent (`__gap`, `__reserved`, etc.)
- [ ] All inheritable contracts in the hierarchy have gaps
- [ ] No contract in the middle of the hierarchy forgot to include a gap

### Diamond Storage Pattern

```solidity
// Each facet uses a unique storage position
library LibDiamond {
    bytes32 constant DIAMOND_STORAGE_POSITION =
        keccak256("diamond.standard.diamond.storage");

    struct DiamondStorage {
        mapping(bytes4 => address) facetAddresses;
        // ...
    }

    function diamondStorage() internal pure returns (DiamondStorage storage ds) {
        bytes32 position = DIAMOND_STORAGE_POSITION;
        assembly { ds.slot := position }
    }
}
```

**Check for diamond patterns:**
- [ ] Each facet uses a unique storage namespace (different keccak256 prefix)
- [ ] No two facets share the same storage position hash
- [ ] Struct layouts within each namespace are upgrade-safe

### Uninitialized Implementation

```
For UUPS and TransparentProxy:
- The implementation contract can be initialized by anyone if not locked
- Check: does the implementation call _disableInitializers() in constructor?
- Check: is there an initializer function protected by initializer modifier?
```

## Output Format

### Storage Layout Report

```markdown
## Storage Layout: VaultV2

### Slot Map

| Slot | Offset | Size | Type | Name | Source |
|------|--------|------|------|------|--------|
| 0 | 0 | 32 | uint256 | totalDeposits | Vault |
| 1 | 0 | 20 | address | owner | Ownable |
| 1 | 20 | 1 | bool | paused | Pausable |
| 1 | 21 | 1 | bool | initialized | Initializable |
| 2 | 0 | 32 | uint256 | feeRate | Vault |
| 3 | 0 | 32 | mapping(address => uint256) | balances | Vault |
| 4-53 | 0 | 1600 | uint256[50] | __gap | Vault |

### Constants & Immutables (no slot)
- `uint256 MAX_FEE` (constant)
- `address token` (immutable)

### Packing Analysis
- Slot 1: 22/32 bytes used (address + 2 bools) — 10 bytes wasted
- Suggestion: add uint80 or smaller type after `initialized` to use remaining space

### Upgrade Safety
- [x] Storage gap present (50 slots)
- [x] Inherited variables maintain original positions
- [ ] WARNING: No gap in Ownable base — future Ownable upgrades may collide
```

### Upgrade Comparison Report

```markdown
## Upgrade Safety: VaultV1 → VaultV2

### Slot Comparison

| Slot | V1 | V2 | Status |
|------|----|----|--------|
| 0 | uint256 totalDeposits | uint256 totalDeposits | ✓ Unchanged |
| 1 | address owner | address owner | ✓ Unchanged |
| 2 | uint256[49] __gap | uint256 newFeeRate | ✓ Uses gap space |
| 3-50 | (part of __gap) | uint256[48] __gap | ✓ Gap shrunk correctly |

### Verdict: SAFE
- No existing variables changed type or position
- New variable uses gap space correctly
- Gap decreased by exactly 1
```

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "Storage layout doesn't matter for non-upgradeable contracts" | Packing affects gas costs. Understanding layout helps with struct optimization and debugging. |
| "The proxy is standard, no need to check" | Standard proxies can still have implementation storage collisions. Verify every time. |
| "Storage gaps are present, so it's safe" | Gaps must be correctly sized, positioned, and maintained across the ENTIRE inheritance chain. |
| "I'll just check the top-level contract" | Storage is inherited. Base contracts' variables come first. Check the full chain. |
| "Diamond storage is isolated by design" | Only if each facet uses a unique namespace. Verify the hash positions don't collide. |
| "The implementation is behind a proxy, so initialization doesn't matter" | Uninitialized implementations can be taken over via direct calls. Always verify _disableInitializers(). |
| "I can eyeball the slot assignments" | Use the tool. Manual counting is error-prone, especially with inheritance and packing. |
| "Constants don't affect storage" | Correct — but verify they ARE actually constants/immutables and not accidentally mutable. |
