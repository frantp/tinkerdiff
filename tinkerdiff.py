from dataclasses import dataclass, field
import difflib
import logging
from typing import Iterator
import xml.etree.ElementTree as ET

from colorama import just_fix_windows_console, Back, Fore, Style
import numpy as np
import scipy as sp

just_fix_windows_console()
logger = logging.getLogger(__name__)
dtype = np.int32


@dataclass(slots=True, frozen=True)
class Element:
    package: str
    value: str = ""
    padtype: str = ""
    internal: bool = field(default=False, compare=False, hash=False)

    def with_value(self, value: str) -> "Element":
        return Element(self.package, value, self.padtype, self.internal)

    def with_padtype(self, padtype: str) -> "Element":
        return Element(self.package, self.value, padtype, self.internal)

    def padless(self) -> "Element":
        return self.with_padtype("")

    def as_internal(self) -> "Element":
        return Element(self.package, self.value, self.padtype, True)

    def shortstr(self) -> str:
        return f"{self.package[:3]}:{self.value[:6]}:{'@' if self.internal else ''}{self.padtype}"

    def __lt__(self, other):
        return str(self) < str(other)

    def __str__(self):
        return (
            f"{self.package}:{self.value}:{'@' if self.internal else ''}{self.padtype}"
        )

    def __repr__(self):
        return str(self)


@dataclass(slots=True, frozen=True)
class Contact:
    element: str
    pad: str = ""

    def with_pad(self, pad) -> "Contact":
        return Contact(self.element, pad)

    def padless(self) -> "Contact":
        return self.with_pad("")

    def __lt__(self, other):
        return str(self) < str(other)

    def __str__(self):
        return f"{self.element}:{self.pad}"

    def __repr__(self):
        return str(self)


def _to(t: type, e):
    return t(**{a: e.attrib[a] for a in e.attrib if a in vars(t)})


@dataclass(slots=True)
class Components:
    padtypes: dict[str, str]
    components: dict[str, list[str]]

    def __str__(self):
        return (
            f"    Pad types:  [{', '.join([f'{k}:{v}' for k, v in self.padtypes.items()])}]\n"
            + f"    Components: [{', '.join([':'.join(c) for c in self.components])}]"
        )

    def __repr__(self):
        return str(self)


@dataclass(slots=True)
class ElementLibrary:
    components: dict[Element, Components] = field(default_factory=lambda: {})

    def load(self, filename: str):
        # TODO: atrib["type"] default nonpolarized
        self.components.update(
            {
                _to(Element, e): Components(
                    {
                        p.attrib["name"]: p.attrib.get("type", p.attrib["name"])
                        for p in e.findall("pad")
                    },
                    {
                        m.attrib["name"]: [
                            c.attrib["pad"] for c in m.findall("contactref")
                        ]
                        for m in e.findall("component")
                    },
                )
                for e in ET.parse(filename).getroot().findall("element")
            }
        )

    def __str__(self):
        return "\n".join([f"{k}:\n{v}" for k, v in self.components.items()])

    def __repr__(self):
        return str(self)


@dataclass(slots=True)
class Schematic:
    packages: dict[str, list[str]]
    elements: dict[Contact, Element]
    signals: dict[str, list[Contact]]

    @staticmethod
    def from_file(filename: str) -> "Schematic | None":
        board = ET.parse(filename).getroot().find("drawing/board")
        if board is None:
            logging.warning("Board not found")
            return None
        packages = {
            p.attrib["name"]: [p.attrib["name"] for p in p.findall("pad")]
            for p in board.findall("libraries/library/packages/package")
        }
        elements = {
            Contact(e.attrib["name"]): _to(Element, e)
            for e in board.findall("elements/element")
        }
        elements = dict(sorted(elements.items()))  # sorted important for initial guess
        signals = {
            s.attrib["name"]: [_to(Contact, c) for c in s.findall("contactref")]
            for s in board.findall("signals/signal")
        }
        return Schematic(packages, elements, signals)

    def types(self):
        return [*{*self.elements.values()}]

    def spread(self, library: ElementLibrary):
        elements = {}
        for c, e in self.elements.items():
            # Try exact element (package + value); then element matching just the package
            ei = library.components.get(
                e, library.components.get(e.with_value(""), None)
            )
            if ei is None:
                package = self.packages[e.package]
                # Single fully-connected component by default
                cm = c.with_pad("@")
                elements[cm] = e.as_internal()
                for cp in package:
                    self.signals[f"{cm}:{cp}"] = [cm, c.with_pad(cp)]
                # Every pad different by default
                for p in package:
                    elements[c.with_pad(p)] = e.with_padtype(p)
            else:
                for k, s in ei.components.items():
                    cm = c.with_pad(f"@{k}")
                    elements[cm] = e.with_padtype(k).as_internal()
                    for cp in s:
                        self.signals[f"{cm}:{cp}"] = [cm, c.with_pad(cp)]
                for p, pt in ei.padtypes.items():
                    # TODO: type="" default nonpolarized
                    elements[c.with_pad(p)] = e.with_padtype(pt)
        self.elements = elements

    def filter(self):
        connected_contacts = {
            ct
            for s in self.signals.values()
            if sum(not self.elements[ct].internal for ct in s) > 1  # At least 2 ext ct
            for ct in s
        }
        self.signals = {
            n: s
            for n, s in self.signals.items()
            if any(ct in connected_contacts for ct in s)  # At least 1 conn ct
        }
        valid_contacts = {ct for s in self.signals.values() for ct in s}
        self.elements = {c: e for c, e in self.elements.items() if c in valid_contacts}

    def contact_matrix(self, strong=1) -> np.ndarray:
        idx = {k: (i, v) for i, (k, v) in enumerate(self.elements.items())}
        N = len(self.elements)
        mat = np.eye(N, dtype=dtype)
        for s in self.signals.values():
            for ci in s:
                # Make it also work with non-spreaded schematic by trying padless contact
                i = idx.get(ci, idx.get(ci.padless()))
                if i is None:
                    raise ValueError("Index not found for contact {ci}")
                for cj in s:
                    j = idx.get(cj, idx.get(cj.padless()))
                    if j is None:
                        raise ValueError("Index not found for contact {cj}")
                    mat[i[0], j[0]] = strong if i[1].internal or j[1].internal else 1
        return mat

    def type_matrix(self, types: list[Element] | None = None) -> np.ndarray:
        if not types:
            types = self.types()
        idx = {v: i for i, v in enumerate(types)}
        N = len(self.elements)
        M = len(types)
        mat = np.zeros((N, M), dtype=dtype)
        for i, v in enumerate(self.elements.values()):
            mat[i, idx[v]] = 1
        return mat

    def ct_matrix(self, N: int = 0, types: list[Element] | None = None) -> np.ndarray:
        mc = self.contact_matrix(N)
        me = self.type_matrix(types) * N
        n = mc.shape[0]
        N = max(N, n)
        M = me.shape[1]
        mat = np.zeros((N + M, N + M), dtype=dtype)
        mat[:n, :n] = mc
        mat[:n, N : N + M] = me
        mat[N : N + M, :n] = me.T
        return mat

    @staticmethod
    def diff(ref: "Schematic", cmp: "Schematic", *, ninit=0) -> Iterator[str]:
        types = sorted([*{*ref.elements.values(), *cmp.elements.values()}])
        N = max(len(ref.elements), len(cmp.elements))
        M = len(types)
        mref = ref.ct_matrix(N, types)
        mcmp = cmp.ct_matrix(N, types)

        # Solve graph matching problem with fixed types
        fixed = np.tile(np.arange(N, N + M, dtype=dtype), (2, 1)).T
        res = max(
            [
                sp.optimize.quadratic_assignment(
                    mref,
                    mcmp,
                    method="faq",
                    options={
                        "maximize": True,
                        "P0": "randomized",
                        "partial_match": fixed,
                    },
                )
                for _ in range(ninit or mref.shape[0])
            ],
            key=lambda x: x.fun,
        )
        logging.info(f"Optimization result:\n{res}")

        # Headers
        mcor = mcmp[res.col_ind, :][:, res.col_ind]
        e_ref = [*ref.elements.items()]
        e_cmp = [*cmp.elements.items()]
        eref = e_ref + [(Contact(""), e) for _, e in e_cmp[len(e_ref) :]]
        ecmp = e_cmp + [(Contact(""), e) for _, e in e_ref[len(e_cmp) :]]
        ecor = [ecmp[i] for i in res.col_ind[: len(ecmp)]]

        # Log visual comparison
        empty = [Element("") for _ in types]
        refstr = matstr(mref, [k for (k, _) in eref], types)
        cmpstr = matstr(mcmp, [k for (k, _) in ecmp], empty)
        corstr = matstr(mcor, [k for (k, _) in ecor], empty, mref)
        allstr = "\n".join(
            f"{r}| {m}| {c}|"
            for r, m, c in zip(
                refstr.splitlines(),
                cmpstr.splitlines(),
                corstr.splitlines(),
            )
        )
        logging.info(f"CT matrices (reference, compared, corrected):\n{allstr}")

        # Return list of text differences
        h = (
            e_ref
            + [ce for ce in ecor[len(e_ref) :]]
            + [(Contact("@TYPE"), e) for e in types]
        )
        return (
            line
            for line in difflib.ndiff(mattext(mref, h, N), mattext(mcor, h, N))
            if line.startswith(("+", "-"))
        )

    def __str__(self):
        p_pad = max(len(k) for k in self.packages.keys())
        e_pad = max(len(str(k)) for k in self.elements.keys())
        s_pad = max(len(k) for k in self.signals.keys())
        return (
            f"""Packages:\n{
                "\n".join(
                    [
                        f"    {k:{p_pad}} [{', '.join(v)}]"
                        for k, v in self.packages.items()
                    ]
                )
            }
            """.strip()
            + "\n"
            + f"""Elements:\n{
                "\n".join(
                    [f"    {str(k):{e_pad}} {v}" for k, v in self.elements.items()]
                )
            }
            """.strip()
            + "\n"
            + f"""Signals: \n{
                "\n".join(
                    [
                        f"    {k:{s_pad}} [{', '.join([f'{c}' for c in v])}]"
                        for k, v in self.signals.items()
                    ]
                )
            }""".strip()
        )


def mattext(mat: np.ndarray, h: list[tuple[Contact, Element]], N) -> list[str]:
    text = []
    for i in range(mat.shape[0]):
        for j in range(min(i, N)):
            if mat[i, j] != 0:
                if (not h[i][1].internal and not h[j][1].internal) or i >= N:
                    # Add non-internal contact errors and any type error
                    text.append(f"{h[i][0]} ({h[i][1]}) - {h[j][0]} ({h[j][1]})")
    return text


def matstr(
    mat: np.ndarray,
    contacts: list[Contact],
    types: list[Element],
    ref: np.ndarray | None = None,
) -> str:
    L = mat.shape[0]
    n = len(contacts)
    M = len(types)
    N = L - M
    names = [str(c) for c in contacts] + [""] * (N - n)
    keys = names + [t.shortstr() for t in types]
    pad = max(len(k) for k in keys)

    hline = f"{'':-^{pad}}-{'':-^{2 * N - 1}} "
    s = ""
    s += hline + "\n"
    ch = {1: "O", 0: "·"}
    for i in range(L):
        if i == N:
            s += hline + "\n"
        s += f"{keys[i]:{pad}} "
        stop = min(i + 1, N)
        for j in range(stop):
            if ref is not None and mat[i, j] != ref[i, j]:
                s += Back.RED
            s += f"{ch.get(mat[i, j], 'X')}"
            if ref is not None:
                s += Style.RESET_ALL
            s += " "
        for j in range(stop, N):
            s += "  "
        s += "\n"
    s += hline
    return s


def diff(
    filename_ref: str, filename_cmp: str, *, spread=True, filter=True, ninit=0
) -> Iterator[str]:
    library = ElementLibrary()
    library.load("library.xml")
    logging.info(f"Library:\n{library}")

    ref = Schematic.from_file(filename_ref)
    if ref is None:
        raise ValueError("Empty schematic (reference)")
    logging.info(f"Schematic (reference):\n{ref}")
    if spread:
        ref.spread(library)
        logging.info(f"Schematic (spreaded) (reference):\n{ref}")
    if filter:
        ref.filter()
        logging.info(f"Schematic (filtered) (reference):\n{ref}")

    cmp = Schematic.from_file(filename_cmp)
    if cmp is None:
        raise ValueError("Empty schematic (compared)")
    logging.info(f"Schematic (compared):\n{cmp}")
    if spread:
        cmp.spread(library)
        logging.info(f"Schematic (spreaded) (compared):\n{cmp}")
    if filter:
        cmp.filter()
        logging.info(f"Schematic (filtered) (compared):\n{cmp}")

    return Schematic.diff(ref, cmp, ninit=ninit)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )

    ref_file = "examples/te.brd"
    cmp_file = "examples/te_mod2.brd"

    diffs = [*diff(ref_file, cmp_file, spread=True, filter=False)]
    if diffs:
        print("Differences:")
        for d in diffs:
            print(
                f"    {Fore.GREEN if d.startswith('+') else Fore.RED}{d}{Style.RESET_ALL}"
            )
    else:
        print("Schematics are equal")
