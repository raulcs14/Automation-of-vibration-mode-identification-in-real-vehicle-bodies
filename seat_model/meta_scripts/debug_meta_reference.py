"""
META API reference script.

Run from inside META post-processor:
    File > Execute Script > debug_meta_reference.py

Sections (comment/uncomment as needed):
    1. Top-level meta modules
    2. Model loading and object inspection
    3. Element iteration — types, subtypes, deck_type
    4. Element attributes and connected nodes
    5. Node object inspection
    6. groups / results quick reference
"""

from meta import models, elements, groups, results, nodes
from config import INPUT_MODAL_DAT

SECTION = 10  # "all" | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# 1. Top-level meta modules
# ---------------------------------------------------------------------------
if SECTION in ("all", 1):
    sep("1. meta module contents")
    import meta
    print(dir(meta))

    sep("1b. Submodules available")
    for name in ["elements", "nodes", "groups", "results", "models",
                 "parts", "materials", "sections", "connections"]:
        try:
            mod = getattr(meta, name)
            print(f"  meta.{name:15s} OK")
        except AttributeError:
            print(f"  meta.{name:15s} NOT FOUND")


# ---------------------------------------------------------------------------
# 2. Model loading and object inspection
# ---------------------------------------------------------------------------
if SECTION in ("all", 2, 3, 4, 5):
    sep("2. Model loading")
    model = models.LoadModel('MetaPost', str(INPUT_MODAL_DAT), 'NASTRAN')
    print(f"  model        : {model}")
    print(f"  model.id     : {model.id}")

    sep("2b. Model methods (filtered)")
    for name in sorted(dir(model)):
        if not name.startswith('_'):
            print(f"  model.{name}")


# ---------------------------------------------------------------------------
# 3. Element types in the model
# ---------------------------------------------------------------------------
if SECTION in ("all", 3, 4):
    sep("3. Element type inventory")
    all_elems = model.get_elements('all')
    print(f"  Total elements: {len(all_elems)}")

    combos = {}
    for e in all_elems:
        key = (e.type, e.subtype, e.get_deck_type())
        combos[key] = combos.get(key, 0) + 1

    print(f"\n  {'type':>6}  {'subtype':>8}  {'deck_type':<20}  {'count':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*20}  {'-'*6}")
    for (t, s, dt), count in sorted(combos.items(), key=lambda x: -x[1]):
        print(f"  {t:>6}  {s:>8}  {dt:<20}  {count:>6}")


# ---------------------------------------------------------------------------
# 4. Element attributes and connected nodes (example: CONM2)
# ---------------------------------------------------------------------------
if SECTION in ("all", 4):
    sep("4. Element object inspection (first CONM2)")

    conm2_elem = None
    for e in model.get_elements('all'):
        if e.get_deck_type() == 'CONM2':
            conm2_elem = e
            break

    if conm2_elem is None:
        print("  No CONM2 elements found in this model.")
    else:
        print(f"  element      : {conm2_elem}")
        print(f"  .id          : {conm2_elem.id}")
        print(f"  .model_id    : {conm2_elem.model_id}")
        print(f"  .part_id     : {conm2_elem.part_id}")
        print(f"  .type        : {conm2_elem.type}")
        print(f"  .subtype     : {conm2_elem.subtype}")
        print(f"  .get_deck_type()    : {conm2_elem.get_deck_type()!r}")
        print(f"  .get_deck_subtype() : {conm2_elem.get_deck_subtype()!r}")

        sep("4b. Element methods")
        for name in sorted(dir(conm2_elem)):
            if not name.startswith('_'):
                print(f"  {name}")

        sep("4c. Connected nodes")
        elem_nodes = conm2_elem.get_nodes()
        print(f"  get_nodes() -> {elem_nodes}")
        for n in elem_nodes:
            print(f"    node.id={n.id}  coords=({n.x:.4f}, {n.y:.4f}, {n.z:.4f})")

        sep("4d. Collecting all CONM2 node IDs")
        conm2_node_ids = sorted({
            e.get_nodes()[0].id
            for e in model.get_elements('all')
            if e.get_deck_type() == 'CONM2' and e.get_nodes()
        })
        print(f"  {len(conm2_node_ids)} CONM2 GRID IDs: {conm2_node_ids}")


# ---------------------------------------------------------------------------
# 5. Node object inspection
# ---------------------------------------------------------------------------
if SECTION in ("all", 5):
    sep("5. Node object inspection (first node)")
    all_nodes = model.get_nodes('all')
    print(f"  Total nodes: {len(all_nodes)}")

    n0 = all_nodes[0]
    print(f"\n  node         : {n0}")
    print(f"  .id          : {n0.id}")
    print(f"  .model_id    : {n0.model_id}")
    print(f"  .x / .y / .z : {n0.x:.4f} / {n0.y:.4f} / {n0.z:.4f}")

    sep("5b. Node methods")
    for name in sorted(dir(n0)):
        if not name.startswith('_'):
            print(f"  {name}")


# ---------------------------------------------------------------------------
# 6. groups / results quick reference
# ---------------------------------------------------------------------------
if SECTION in ("all", 6):
    sep("6. groups module functions")
    for name in sorted(dir(groups)):
        if not name.startswith('_'):
            print(f"  groups.{name}")

    sep("6b. results module functions (first 30)")
    for name in sorted(dir(results))[:30]:
        if not name.startswith('_'):
            print(f"  results.{name}")

print("\nDone.")

# ---------------------------------------------------------------------------
# 10. meta.parts — Parts(), PartOfElement(), CommentsOfPart(), NameOfPart
# ---------------------------------------------------------------------------
if SECTION in ("all", 10):
    import meta
    model = models.LoadModel('MetaPost', str(INPUT_MODAL_DAT), 'NASTRAN')
    mid = model.id

    sep("10a. meta.parts.Parts(model_id) — all parts")
    try:
        all_parts = meta.parts.Parts(mid)
        print(f"  Parts({mid}) -> {len(all_parts)} parts")
        if all_parts:
            p0 = all_parts[0]
            print(f"  first part type : {type(p0)}")
            print(f"  first part attrs: {[a for a in dir(p0) if not a.startswith('_')]}")
            for a in ['id', 'name', 'Name', 'pid', 'comment', 'type']:
                if hasattr(p0, a):
                    print(f"    .{a} = {getattr(p0, a)}")
    except Exception as e:
        print(f"  Parts() error: {e}")

    sep("10b. meta.parts.CommentsOfPart — first 5 parts")
    try:
        all_parts = meta.parts.Parts(mid)
        for p in all_parts[:5]:
            try:
                comment = meta.parts.CommentsOfPart(mid, p)
                name_attr = getattr(p, 'name', getattr(p, 'Name', None))
                pid_attr  = getattr(p, 'id',   getattr(p, 'pid',  '?'))
                print(f"  part id={pid_attr}  name={name_attr}  comment={comment}")
            except Exception as e:
                print(f"  CommentsOfPart error: {e}")
    except Exception as e:
        print(f"  error: {e}")

    sep("10c. meta.parts.PartOfElement — first CQUAD4")
    try:
        all_elems = model.get_elements('all')
        quad = next((e for e in all_elems if e.get_deck_type() == 'CQUAD4'), None)
        if quad:
            part = meta.parts.PartOfElement(mid, quad)
            print(f"  PartOfElement(quad id={quad.id}) -> {part}  type={type(part)}")
            if part is not None:
                print(f"  part attrs: {[a for a in dir(part) if not a.startswith('_')]}")
                for a in ['id', 'name', 'Name', 'pid', 'comment']:
                    if hasattr(part, a):
                        print(f"    .{a} = {getattr(part, a)}")
                # Try getting name via meta.parts functions
                for fn in ['CommentsOfPart', 'AttributeOfPart', 'AttributesOfPart']:
                    if hasattr(meta.parts, fn):
                        try:
                            r = getattr(meta.parts, fn)(mid, part)
                            print(f"  meta.parts.{fn}(part) -> {r}")
                        except Exception as e:
                            print(f"  meta.parts.{fn} error: {e}")
    except Exception as e:
        print(f"  PartOfElement error: {e}")

    sep("10d. meta.parts.PartsByType — shell parts")
    try:
        shell_parts = meta.parts.PartsByType(mid, 'PSHELL')
        print(f"  PartsByType(PSHELL) -> {len(shell_parts)} parts")
        for p in shell_parts[:3]:
            pid_attr = getattr(p, 'id', getattr(p, 'pid', '?'))
            comment  = meta.parts.CommentsOfPart(mid, p)
            print(f"    id={pid_attr}  comment={comment}")
    except Exception as e:
        print(f"  PartsByType error: {e}")

# ---------------------------------------------------------------------------
# 9. Explore meta.parts and meta.base hierarchy
# ---------------------------------------------------------------------------
if SECTION in ("all", 9):
    import meta
    import meta.base as base
    model = models.LoadModel('MetaPost', str(INPUT_MODAL_DAT), 'NASTRAN')

    sep("9a. meta.parts — full dir")
    print(dir(meta.parts))

    sep("9b. meta.parts — try common calls")
    for fn_name in ['GetAll', 'GetParts', 'GetPart', 'GetEntities',
                    'GetList', 'CollectParts']:
        if hasattr(meta.parts, fn_name):
            try:
                fn = getattr(meta.parts, fn_name)
                r  = fn(model.id)
                print(f"  meta.parts.{fn_name}({model.id}) -> {len(r)} items")
                if r:
                    item = r[0]
                    attrs = [a for a in dir(item) if not a.startswith('_')]
                    print(f"    first item: type={type(item)}  attrs={attrs}")
                    for a in ['id', 'name', 'Name', 'pid', 'comment', 'include_id']:
                        if hasattr(item, a):
                            print(f"      .{a} = {getattr(item, a)}")
            except Exception as e:
                print(f"  meta.parts.{fn_name}({model.id}) -> ERROR: {e}")
        else:
            print(f"  meta.parts.{fn_name:20s} NOT FOUND")

    sep("9c. meta.base.GetRootsList — hierarchy root")
    try:
        roots = base.GetRootsList()
        print(f"  GetRootsList() -> {roots}")
    except Exception as e:
        print(f"  GetRootsList() error: {e}")

    sep("9d. meta.base.GetObjectHierarchyIds — walk hierarchy")
    try:
        roots = base.GetRootsList()
        if roots:
            root_id = roots[0]
            children = base.GetObjectHierarchyIds(root_id)
            print(f"  root_id={root_id}  children={children[:10]}")
    except Exception as e:
        print(f"  GetObjectHierarchyIds error: {e}")

    sep("9e. meta.base.GetDMRootsList / GetAllItemsInDM")
    for fn_name in ['GetDMRootsList', 'GetAllItemsInDM', 'GetDMRoot']:
        try:
            r = getattr(base, fn_name)()
            print(f"  base.{fn_name}() -> {r}")
        except Exception as e:
            print(f"  base.{fn_name}() -> ERROR: {e}")

# ---------------------------------------------------------------------------
# 7. Part/group hierarchy exploration
# ---------------------------------------------------------------------------
if SECTION in ("all", 7):
    sep("7. meta module — parts/groups submodules")
    import meta
    for name in ["parts", "groups", "collections", "includes", "sets"]:
        try:
            mod = getattr(meta, name)
            print(f"  meta.{name:15s} OK  ->  {dir(mod)}")
        except AttributeError:
            print(f"  meta.{name:15s} NOT FOUND")

    sep("7b. model.get_parts / model.get_groups / model.get_includes")
    model = models.LoadModel('MetaPost', str(INPUT_MODAL_DAT), 'NASTRAN')
    for method in ["get_parts", "get_groups", "get_includes",
                   "get_collections", "get_sets"]:
        if hasattr(model, method):
            try:
                result = getattr(model, method)('all')
                print(f"  model.{method}('all') -> {len(result)} items")
                if result:
                    item = result[0]
                    print(f"    first item type : {type(item)}")
                    print(f"    first item attrs: {[a for a in dir(item) if not a.startswith('_')]}")
                    for attr in ["id", "name", "Name", "pid", "mid"]:
                        if hasattr(item, attr):
                            print(f"      .{attr} = {getattr(item, attr)}")
            except Exception as e:
                print(f"  model.{method} ERROR: {e}")
        else:
            print(f"  model.{method:30s} NOT FOUND")

    sep("7c. First element — all non-private attributes")
    all_elems = model.get_elements('all')
    structural = [e for e in all_elems if e.get_deck_type() in
                  ('CQUAD4', 'CTRIA3', 'CBAR', 'CBEAM')]
    if structural:
        e = structural[0]
        print(f"  Element type: {e.get_deck_type()}  id={e.id}")
        for attr in sorted(dir(e)):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(e, attr)
                if not callable(val):
                    print(f"    .{attr} = {val}")
            except Exception:
                pass

    sep("7d. model — ALL methods and attributes")
    for attr in sorted(dir(model)):
        if not attr.startswith('_'):
            print(f"  model.{attr}")


# ---------------------------------------------------------------------------
# 8. Try every plausible way to get named parts from META
# ---------------------------------------------------------------------------
if SECTION in ("all", 8):
    import meta
    model = models.LoadModel('MetaPost', str(INPUT_MODAL_DAT), 'NASTRAN')

    sep("8a. meta top-level dir")
    print([x for x in dir(meta) if not x.startswith('_')])

    sep("8b. Try meta.base module (ANSA scripting uses meta.base)")
    try:
        import meta.base as base
        print(f"  meta.base OK: {[x for x in dir(base) if not x.startswith('_')]}")
    except Exception as e:
        print(f"  meta.base: {e}")

    sep("8c. model.get_include_ids / get_part_ids")
    for method_name in ['get_include_ids', 'get_part_ids', 'get_pid_ids',
                        'get_mid_ids', 'GetIncludes', 'GetParts']:
        if hasattr(model, method_name):
            try:
                r = getattr(model, method_name)()
                print(f"  model.{method_name}() -> {r}")
            except Exception as e:
                print(f"  model.{method_name}() error: {e}")
        else:
            print(f"  model.{method_name:30s} NOT FOUND")

    sep("8d. First CQUAD4 element — try get_property, get_include")
    all_elems = model.get_elements('all')
    quad = next((e for e in all_elems if e.get_deck_type() == 'CQUAD4'), None)
    if quad:
        print(f"  quad id={quad.id}  part_id={quad.part_id}")
        for method_name in ['get_property', 'get_include', 'get_part',
                            'GetProperty', 'GetInclude', 'GetPart']:
            if hasattr(quad, method_name):
                try:
                    r = getattr(quad, method_name)()
                    print(f"  elem.{method_name}() -> {r}  type={type(r)}")
                    if r is not None:
                        print(f"    attrs: {[a for a in dir(r) if not a.startswith('_')]}")
                        for a in ['id', 'name', 'Name', 'pid', 'comment']:
                            if hasattr(r, a):
                                print(f"      .{a} = {getattr(r, a)}")
                except Exception as e:
                    print(f"  elem.{method_name}() error: {e}")
            else:
                print(f"  elem.{method_name:25s} NOT FOUND")
