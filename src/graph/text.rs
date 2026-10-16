//! A dependency-free, line-oriented text format for [`Graph`].
//!
//! Graphs need to be saved, diffed, hand-edited and loaded by tools that don't
//! link this crate (shell scripts, the example runner, the editor). A tiny
//! line format serves all of that without pulling in serde: one statement per
//! line, so a diff of two graph files reads as a diff of nodes, edges and
//! node positions.
//!
//! ```text
//! pipe-graph 2
//! # comments may appear anywhere
//! node split split channels=3
//! node merge merge channels=3
//! node "my node" crop h=2 w=2 x=0 y=0
//! edge split.out0 -> merge.in0
//! edge "my node".out -> split.in
//! pos merge 260 0
//! pos split 0 -12.5
//! ```
//!
//! # Versions
//!
//! The first statement of a file may be a version header, `pipe-graph <n>`.
//!
//! * **v1** — `node` and `edge` statements only. A file without a header is
//!   read as v1, so graphs written before the header existed (which start with
//!   a `# pipe-graph text format v1` *comment*) keep loading unchanged. An
//!   explicit `pipe-graph 1` header is accepted too.
//! * **v2** — adds `pos` statements: the canvas layout of an editor. The layout
//!   is optional (a v2 file without `pos` lines is fine) and never affects
//!   how the graph runs.
//!
//! The writer always emits the current version ([`FORMAT_VERSION`]). A header
//! naming a newer version is rejected with
//! [`ParseErrorKind::UnsupportedVersion`] rather than parsed on a best-effort
//! basis: a newer file may hold statements whose meaning this reader cannot
//! know, and silently dropping them would lose data on the next save. `pos` in
//! a v1 file is an error for the same reason in reverse — a file that claims
//! v1 should be readable by a v1 reader.
//!
//! # Grammar
//!
//! ```text
//! file      = { line "\n" } [ line ]       ; a trailing "\r" on a line is ignored
//! line      = ws* [ statement ] ws* [ comment ]
//! comment   = "#" { any char }             ; only outside a quoted atom
//! statement = header | node | edge | pos
//! header    = "pipe-graph" digits          ; only as the first statement
//! node      = "node" atom atom { param }   ; id, kind, params
//! param     = atom "=" atom                ; key=value
//! edge      = "edge" endpoint "->" endpoint
//! endpoint  = atom "." atom                ; node.port
//! pos       = "pos" atom number number     ; node id, x, y (v2 only)
//! number    = finite f32 as accepted by Rust's `str::parse::<f32>`, written
//!             without whitespace or quotes (e.g. 12, -0.5, 1e-3)
//! atom      = bare | quoted
//! bare      = 1*( any char except whitespace, '"', '#', '=', '.', '\' )
//! quoted    = '"' { char | escape } '"'
//! escape    = '\\' | '\"' | '\n' | '\r' | '\t'
//! ws        = " " | "\t"
//! ```
//!
//! * Whitespace separates atoms; it is optional around `=` and `.` but
//!   **required** around `->`, since `-` and `>` are ordinary bare characters
//!   (`a.out->b.in` lexes `out->b` as one atom).
//! * `pipe-graph`, `node`, `edge`, `pos` and `->` must be written bare: a
//!   quoted `"->"` is a port/id named `->`, not the arrow. Inside quotes every
//!   character except `"` and `\` is literal, so ids, kinds, ports, keys and
//!   values may contain spaces, `=`, `.`, `#` or non-ASCII text.
//! * A quoted atom must be followed by whitespace, `=`, `.`, a comment or the
//!   end of the line: `"a"b` is rejected rather than silently read as two atoms.
//! * Statements other than the header may appear in any order; an edge or
//!   `pos` may name a node declared on a later line. Duplicate node ids,
//!   duplicate param keys on one node, a second `pos` for one node, edges or
//!   `pos` lines naming undeclared nodes, malformed or non-finite numbers and
//!   malformed lines are reported as a [`ParseError`] carrying the 1-based
//!   line number.
//! * `pos` coordinates are canvas coordinates: +x right, +y down (the editor's
//!   `Layout` convention). They are plain data here — this module does not
//!   depend on the editor.
//!
//! # Determinism
//!
//! [`GraphDocument::to_text`] (and [`Graph::to_text`], which writes a document
//! with no positions) writes the `pipe-graph 2` header, then nodes sorted by id
//! (params sorted by key), then edges in ascending [`EdgeId`] order — i.e. the
//! order they were connected — then `pos` lines sorted by node id. It emits an
//! atom bare only when it is non-empty and consists solely of ASCII
//! alphanumerics and `_ - + / : @`; everything else is quoted. Coordinates are
//! printed with Rust's `{}` formatting for `f32`, the shortest decimal that
//! parses back to the identical value, so positions round-trip bit-exactly.
//! No comments are written. The output is therefore a pure function of the
//! document, and re-writing a parsed file that this writer produced
//! reproduces it byte-for-byte.
//!
//! Edge ids are **not** serialized: [`Graph::from_text`] re-connects edges in
//! file order, so a re-loaded graph gets fresh ids `0..n` in the same relative
//! order as the original.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;

use super::{Connection, EdgeId, Graph, GraphError, NodeId, NodeSpec, Params, PortId};

/// The format version [`GraphDocument::to_text`] writes, and the newest one
/// the reader accepts.
pub const FORMAT_VERSION: u32 = 2;

/// The keyword of the version header line (`pipe-graph <n>`).
const HEADER_KEYWORD: &str = "pipe-graph";

/// Why a line of graph text was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseErrorKind {
    /// A `"` was opened but the line ended before it was closed.
    UnterminatedString,
    /// `\` followed by a character that is not a recognised escape (or by the
    /// end of the line).
    BadEscape(Option<char>),
    /// A stray character where none is allowed (e.g. `\` outside quotes, or an
    /// atom glued to a closing quote).
    UnexpectedChar(char),
    /// The first word of a statement was not a known statement keyword.
    UnknownStatement(String),
    /// The statement ended or continued with something other than `expected`.
    Expected(&'static str),
    /// Two `node` statements declared the same id.
    DuplicateNodeId(String),
    /// One `node` statement repeated a param key.
    DuplicateParam(String),
    /// An `edge` names a node that no `node` statement declares.
    UnknownNode(String),
    /// A `pipe-graph` header appeared after another statement (or twice).
    MisplacedHeader,
    /// The header's version is not a positive integer.
    BadVersion(String),
    /// The header names a version newer than [`FORMAT_VERSION`].
    UnsupportedVersion(u32),
    /// A statement that only exists from `version` on, in an older file
    /// (e.g. `pos` in a file with no `pipe-graph 2` header).
    RequiresVersion {
        statement: &'static str,
        version: u32,
    },
    /// A coordinate that is not a finite `f32`.
    BadNumber(String),
    /// Two `pos` statements for the same node.
    DuplicatePosition(String),
    /// A `pos` names a node that no `node` statement declares.
    UnknownPositionNode(String),
}

/// A parse failure, located at a 1-based line number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub kind: ParseErrorKind,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: ", self.line)?;
        match &self.kind {
            ParseErrorKind::UnterminatedString => write!(f, "unterminated quoted string"),
            ParseErrorKind::BadEscape(Some(c)) => write!(f, "unknown escape '\\{c}'"),
            ParseErrorKind::BadEscape(None) => write!(f, "'\\' at end of line"),
            ParseErrorKind::UnexpectedChar(c) => write!(f, "unexpected character '{c}'"),
            ParseErrorKind::UnknownStatement(s) => {
                write!(
                    f,
                    "unknown statement '{s}' (expected 'node', 'edge' or 'pos')"
                )
            }
            ParseErrorKind::Expected(what) => write!(f, "expected {what}"),
            ParseErrorKind::DuplicateNodeId(id) => write!(f, "duplicate node id '{id}'"),
            ParseErrorKind::DuplicateParam(k) => write!(f, "duplicate parameter '{k}'"),
            ParseErrorKind::UnknownNode(id) => write!(f, "edge references unknown node '{id}'"),
            ParseErrorKind::MisplacedHeader => write!(
                f,
                "'{HEADER_KEYWORD}' version header must be the first statement"
            ),
            ParseErrorKind::BadVersion(v) => write!(f, "bad format version '{v}'"),
            ParseErrorKind::UnsupportedVersion(v) => write!(
                f,
                "format version {v} is newer than this reader supports (max {FORMAT_VERSION})"
            ),
            ParseErrorKind::RequiresVersion { statement, version } => write!(
                f,
                "'{statement}' requires a '{HEADER_KEYWORD} {version}' header"
            ),
            ParseErrorKind::BadNumber(n) => write!(f, "bad coordinate '{n}'"),
            ParseErrorKind::DuplicatePosition(id) => {
                write!(f, "duplicate position for node '{id}'")
            }
            ParseErrorKind::UnknownPositionNode(id) => {
                write!(f, "pos references unknown node '{id}'")
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// A graph plus its canvas layout: everything a saved editor file holds.
///
/// Positions are plain `(x, y)` canvas coordinates (+x right, +y down) so the
/// `graph` module stays independent of the editor; the editor converts to and
/// from its `Layout` (see `LiveSession::to_document` / `from_document`).
#[derive(Debug, Clone, Default)]
pub struct GraphDocument {
    pub graph: Graph,
    /// Node positions. Need not cover every node: a loader auto-places the
    /// rest. When writing, entries for nodes not in `graph` and non-finite
    /// coordinates are skipped, since neither could be read back.
    pub positions: HashMap<NodeId, (f32, f32)>,
}

impl GraphDocument {
    /// A document with no layout.
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            positions: HashMap::new(),
        }
    }

    /// Serialize to the text format (see the module docs in
    /// `src/graph/text.rs`), always as version [`FORMAT_VERSION`].
    pub fn to_text(&self) -> String {
        write_text(&self.graph, &self.positions)
    }

    /// Parse the text format (see the module docs in `src/graph/text.rs`),
    /// any version up to [`FORMAT_VERSION`]. A v1 file yields an empty
    /// `positions` map.
    pub fn from_text(text: &str) -> Result<GraphDocument, ParseError> {
        let mut graph = Graph::new();
        // Edges and positions are resolved after all nodes are known, so
        // files may declare nodes after the statements that use them.
        let mut edges: Vec<(usize, Endpoint, Endpoint)> = Vec::new();
        let mut positions: HashMap<NodeId, (f32, f32)> = HashMap::new();
        let mut position_lines: Vec<(usize, NodeId)> = Vec::new();
        // `None` until the first statement; a header must be that statement.
        let mut version: Option<u32> = None;

        for (idx, raw) in text.split('\n').enumerate() {
            let line = idx + 1;
            let raw = raw.strip_suffix('\r').unwrap_or(raw);
            let err = |kind| ParseError { line, kind };
            let toks = lex(raw).map_err(err)?;
            let mut p = Cursor {
                toks: &toks,
                pos: 0,
            };

            let Some(head) = p.next() else { continue };
            let keyword = match head {
                Tok::Atom {
                    text,
                    quoted: false,
                } => Some(text.as_str()),
                _ => None,
            };

            if keyword == Some(HEADER_KEYWORD) {
                if version.is_some() {
                    return Err(err(ParseErrorKind::MisplacedHeader));
                }
                // Read the version like a number (`2.0` is one bad version,
                // not `2` followed by junk); quoted versions are rejected.
                let v = p.word("format version").map_err(err)?;
                let n = match v.parse::<u32>() {
                    Ok(n) if n > 0 && !v.starts_with('+') => n,
                    _ => return Err(err(ParseErrorKind::BadVersion(v))),
                };
                if n > FORMAT_VERSION {
                    return Err(err(ParseErrorKind::UnsupportedVersion(n)));
                }
                p.end("end of line after version header").map_err(err)?;
                version = Some(n);
                continue;
            }
            // First statement without a header: a legacy (v1) file.
            let file_version = *version.get_or_insert(1);

            match keyword {
                Some("node") => {
                    let id = p.atom("node id").map_err(err)?;
                    let kind = p.atom("node kind").map_err(err)?;
                    let mut params = Params::new();
                    while p.peek().is_some() {
                        let key = p.atom("parameter key").map_err(err)?;
                        p.expect(&Tok::Eq, "'=' after parameter key").map_err(err)?;
                        let value = p.atom("parameter value").map_err(err)?;
                        if params.contains_key(&key) {
                            return Err(err(ParseErrorKind::DuplicateParam(key)));
                        }
                        params.insert(key, value);
                    }
                    let spec = NodeSpec {
                        id: NodeId(id),
                        kind,
                        params,
                    };
                    graph.add_node(spec).map_err(|e| match e {
                        GraphError::DuplicateNodeId(id) => err(ParseErrorKind::DuplicateNodeId(id)),
                        // add_node only ever reports duplicates.
                        _ => unreachable!("add_node returned {e:?}"),
                    })?;
                }
                Some("edge") => {
                    let from = p.endpoint().map_err(err)?;
                    match p.next() {
                        Some(Tok::Atom {
                            text,
                            quoted: false,
                        }) if text == "->" => {}
                        _ => return Err(err(ParseErrorKind::Expected("'->' between endpoints"))),
                    }
                    let to = p.endpoint().map_err(err)?;
                    p.end("end of line after edge").map_err(err)?;
                    edges.push((line, from, to));
                }
                Some("pos") => {
                    if file_version < 2 {
                        return Err(err(ParseErrorKind::RequiresVersion {
                            statement: "pos",
                            version: 2,
                        }));
                    }
                    let id = NodeId(p.atom("node id after 'pos'").map_err(err)?);
                    let x = p.number("x coordinate").map_err(err)?;
                    let y = p.number("y coordinate").map_err(err)?;
                    p.end("end of line after pos").map_err(err)?;
                    match positions.entry(id.clone()) {
                        Entry::Occupied(_) => {
                            return Err(err(ParseErrorKind::DuplicatePosition(id.0)));
                        }
                        Entry::Vacant(slot) => {
                            slot.insert((x, y));
                        }
                    }
                    position_lines.push((line, id));
                }
                _ => match head {
                    Tok::Atom { text, .. } => {
                        return Err(err(ParseErrorKind::UnknownStatement(text.clone())));
                    }
                    _ => return Err(err(ParseErrorKind::Expected("'node', 'edge' or 'pos'"))),
                },
            }
        }

        // Both checks below report dangling node references; whichever comes
        // first in the file is reported, so errors read top to bottom.
        let pos_error = position_lines
            .into_iter()
            .find(|(_, id)| !graph.nodes.contains_key(id))
            .map(|(line, id)| ParseError {
                line,
                kind: ParseErrorKind::UnknownPositionNode(id.0),
            });
        for (line, from, to) in edges {
            if let Err(e) = graph.connect(from, to) {
                let edge_error = ParseError {
                    line,
                    kind: match e {
                        GraphError::MissingNode(id) => ParseErrorKind::UnknownNode(id),
                        _ => unreachable!("connect returned {e:?}"),
                    },
                };
                return Err(match pos_error {
                    Some(p) if p.line < line => p,
                    _ => edge_error,
                });
            }
        }
        if let Some(e) = pos_error {
            return Err(e);
        }
        Ok(GraphDocument { graph, positions })
    }
}

impl std::str::FromStr for GraphDocument {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        GraphDocument::from_text(s)
    }
}

impl Graph {
    /// Serialize to the text format (see the module docs in
    /// `src/graph/text.rs`) as a [`GraphDocument`] with no layout: the
    /// current-version header, nodes and edges, no `pos` lines.
    ///
    /// Output is deterministic: nodes by id, params by key, edges by
    /// [`EdgeId`].
    pub fn to_text(&self) -> String {
        write_text(self, &HashMap::new())
    }

    /// Parse the text format (see the module docs in `src/graph/text.rs`)
    /// into a new graph, discarding any layout. `pos` lines are still
    /// validated, so a file this accepts is one [`GraphDocument::from_text`]
    /// accepts too.
    ///
    /// Edges are connected in file order and receive fresh [`EdgeId`]s.
    pub fn from_text(text: &str) -> Result<Graph, ParseError> {
        GraphDocument::from_text(text).map(|doc| doc.graph)
    }

    /// Edges in ascending [`EdgeId`] order (i.e. connection order) — the
    /// stable order used by [`Graph::to_text`].
    pub fn edges_in_order(&self) -> Vec<(&EdgeId, &Connection)> {
        let mut edges: Vec<(&EdgeId, &Connection)> = self.edges.iter().collect();
        edges.sort_by_key(|(id, _)| id.0);
        edges
    }
}

impl std::str::FromStr for Graph {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Graph::from_text(s)
    }
}

/// The writer behind [`GraphDocument::to_text`] and [`Graph::to_text`]; takes
/// borrowed parts so a bare graph is written without cloning it.
fn write_text(graph: &Graph, positions: &HashMap<NodeId, (f32, f32)>) -> String {
    let mut out = format!("{HEADER_KEYWORD} {FORMAT_VERSION}\n");

    let mut nodes: Vec<&NodeSpec> = graph.nodes.values().collect();
    nodes.sort_by(|a, b| a.id.0.cmp(&b.id.0));
    for spec in nodes {
        out.push_str("node ");
        write_atom(&mut out, &spec.id.0);
        out.push(' ');
        write_atom(&mut out, &spec.kind);
        let mut params: Vec<(&String, &String)> = spec.params.iter().collect();
        params.sort();
        for (k, v) in params {
            out.push(' ');
            write_atom(&mut out, k);
            out.push('=');
            write_atom(&mut out, v);
        }
        out.push('\n');
    }

    for (_, conn) in graph.edges_in_order() {
        out.push_str("edge ");
        write_atom(&mut out, &conn.from.0.0);
        out.push('.');
        write_atom(&mut out, &conn.from.1.0);
        out.push_str(" -> ");
        write_atom(&mut out, &conn.to.0.0);
        out.push('.');
        write_atom(&mut out, &conn.to.1.0);
        out.push('\n');
    }

    let mut sorted: Vec<(&NodeId, &(f32, f32))> = positions
        .iter()
        .filter(|(id, (x, y))| graph.nodes.contains_key(*id) && x.is_finite() && y.is_finite())
        .collect();
    sorted.sort_by(|a, b| a.0.0.cmp(&b.0.0));
    for (id, (x, y)) in sorted {
        out.push_str("pos ");
        write_atom(&mut out, &id.0);
        // `{}` on f32 is the shortest representation that parses back to
        // the same bits, and never uses exponent notation or quotes.
        out.push_str(&format!(" {x} {y}\n"));
    }
    out
}

/// Characters the writer may leave unquoted. Deliberately narrower than what
/// the reader accepts bare, so the writer never depends on lexer corner cases.
fn is_safe_bare(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '+' | '/' | ':' | '@')
}

fn write_atom(out: &mut String, s: &str) {
    if !s.is_empty() && s.chars().all(is_safe_bare) {
        out.push_str(s);
        return;
    }
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A `(node, port)` pair as written in an `edge` statement.
type Endpoint = (NodeId, PortId);

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Atom { text: String, quoted: bool },
    Eq,
    Dot,
}

/// A token plus its byte span in the line. Spans let the parser tell `1.5`
/// (three glued tokens: `1`, `.`, `5`) from `1 . 5` — the lexer treats `.` as
/// punctuation everywhere, and only `pos` coordinates need to re-join it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Lexed {
    tok: Tok,
    start: usize,
    end: usize,
}

/// Characters that end a bare atom.
fn is_delim(c: char) -> bool {
    c == ' ' || c == '\t' || matches!(c, '"' | '#' | '=' | '.' | '\\')
}

fn lex(line: &str) -> Result<Vec<Lexed>, ParseErrorKind> {
    let mut toks = Vec::new();
    let mut chars = line.char_indices().peekable();
    // Byte offset just past the last consumed char.
    let offset = |chars: &mut std::iter::Peekable<std::str::CharIndices>| {
        chars.peek().map_or(line.len(), |&(i, _)| i)
    };
    while let Some(&(start, c)) = chars.peek() {
        match c {
            ' ' | '\t' => {
                chars.next();
            }
            '#' => break,
            '=' | '.' => {
                chars.next();
                toks.push(Lexed {
                    tok: if c == '=' { Tok::Eq } else { Tok::Dot },
                    start,
                    end: start + 1,
                });
            }
            '\\' => return Err(ParseErrorKind::UnexpectedChar('\\')),
            '"' => {
                chars.next();
                let mut text = String::new();
                loop {
                    match chars.next().map(|(_, c)| c) {
                        None => return Err(ParseErrorKind::UnterminatedString),
                        Some('"') => break,
                        Some('\\') => match chars.next().map(|(_, c)| c) {
                            Some('"') => text.push('"'),
                            Some('\\') => text.push('\\'),
                            Some('n') => text.push('\n'),
                            Some('r') => text.push('\r'),
                            Some('t') => text.push('\t'),
                            other => return Err(ParseErrorKind::BadEscape(other)),
                        },
                        Some(c) => text.push(c),
                    }
                }
                // Forbid `"a"b` / `"a""b"`: two atoms with no separator are
                // almost certainly a typo, not two tokens.
                if let Some(&(_, next)) = chars.peek()
                    && !matches!(next, ' ' | '\t' | '=' | '.' | '#')
                {
                    return Err(ParseErrorKind::UnexpectedChar(next));
                }
                let end = offset(&mut chars);
                toks.push(Lexed {
                    tok: Tok::Atom { text, quoted: true },
                    start,
                    end,
                });
            }
            _ => {
                let mut text = String::new();
                while let Some(&(_, c)) = chars.peek() {
                    if is_delim(c) {
                        break;
                    }
                    text.push(c);
                    chars.next();
                }
                let end = offset(&mut chars);
                toks.push(Lexed {
                    tok: Tok::Atom {
                        text,
                        quoted: false,
                    },
                    start,
                    end,
                });
            }
        }
    }
    Ok(toks)
}

struct Cursor<'a> {
    toks: &'a [Lexed],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn peek(&self) -> Option<&'a Tok> {
        self.toks.get(self.pos).map(|l| &l.tok)
    }

    fn next(&mut self) -> Option<&'a Tok> {
        let t = self.peek();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn atom(&mut self, what: &'static str) -> Result<String, ParseErrorKind> {
        match self.next() {
            Some(Tok::Atom { text, .. }) => Ok(text.clone()),
            _ => Err(ParseErrorKind::Expected(what)),
        }
    }

    fn expect(&mut self, tok: &Tok, what: &'static str) -> Result<(), ParseErrorKind> {
        match self.next() {
            Some(t) if t == tok => Ok(()),
            _ => Err(ParseErrorKind::Expected(what)),
        }
    }

    /// Require that the statement has no further tokens.
    fn end(&self, what: &'static str) -> Result<(), ParseErrorKind> {
        match self.peek() {
            None => Ok(()),
            Some(_) => Err(ParseErrorKind::Expected(what)),
        }
    }

    fn endpoint(&mut self) -> Result<Endpoint, ParseErrorKind> {
        let node = self.atom("node id in edge endpoint")?;
        self.expect(&Tok::Dot, "'.' between node and port")?;
        let port = self.atom("port name in edge endpoint")?;
        Ok((NodeId(node), PortId(port)))
    }

    /// A numeric-looking word: the longest run of *adjacent* bare atoms and
    /// dots (the lexer splits `-1.5e3` into `-1`, `.`, `5e3`), re-joined.
    /// Whitespace ends the word, so `1 . 5` is not `1.5`, and the word must
    /// itself be separated from the token before it, so `pos a.5 3` is not
    /// node `a` at x = 0.5.
    fn word(&mut self, what: &'static str) -> Result<String, ParseErrorKind> {
        if let Some(prev) = self.pos.checked_sub(1).and_then(|i| self.toks.get(i))
            && let Some(cur) = self.toks.get(self.pos)
            && prev.end == cur.start
        {
            return Err(ParseErrorKind::Expected(what));
        }
        let mut text = String::new();
        let mut last_end = None;
        while let Some(l) = self.toks.get(self.pos) {
            if last_end.is_some_and(|end| end != l.start) {
                break;
            }
            match &l.tok {
                Tok::Atom {
                    text: t,
                    quoted: false,
                } => text.push_str(t),
                Tok::Dot => text.push('.'),
                _ => break,
            }
            last_end = Some(l.end);
            self.pos += 1;
        }
        if text.is_empty() {
            return Err(ParseErrorKind::Expected(what));
        }
        Ok(text)
    }

    /// A finite `f32`, read as a [`Cursor::word`].
    fn number(&mut self, what: &'static str) -> Result<f32, ParseErrorKind> {
        let text = self.word(what)?;
        // `parse` accepts "inf"/"NaN" and overflows to infinity; neither is a
        // meaningful canvas coordinate, and writing one back would not be
        // readable by every tool, so only finite values are accepted.
        match text.parse::<f32>() {
            Ok(v) if v.is_finite() => Ok(v),
            _ => Err(ParseErrorKind::BadNumber(text)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atoms(line: &str) -> Vec<String> {
        lex(line)
            .unwrap()
            .into_iter()
            .map(|l| match l.tok {
                Tok::Atom { text, .. } => text,
                Tok::Eq => "=".into(),
                Tok::Dot => ".".into(),
            })
            .collect()
    }

    fn number(line: &str) -> Result<f32, ParseErrorKind> {
        let toks = lex(line).unwrap();
        let mut c = Cursor {
            toks: &toks,
            pos: 0,
        };
        let v = c.number("number")?;
        c.end("end")?;
        Ok(v)
    }

    #[test]
    fn lexes_bare_quoted_and_punctuation() {
        assert_eq!(
            atoms(r#"node "a b" k=v "x=y"="1.5" # tail"#),
            ["node", "a b", "k", "=", "v", "x=y", "=", "1.5"]
        );
        assert_eq!(
            atoms("edge a.out -> b.in"),
            ["edge", "a", ".", "out", "->", "b", ".", "in"]
        );
    }

    #[test]
    fn lexes_spans_in_bytes() {
        let toks = lex(r#" é."x" "#).unwrap();
        let spans: Vec<(usize, usize)> = toks.iter().map(|l| (l.start, l.end)).collect();
        assert_eq!(spans, [(1, 3), (3, 4), (4, 7)]);
    }

    #[test]
    fn lexes_escapes() {
        assert_eq!(atoms(r#""q\"\\\n\t\r""#), ["q\"\\\n\t\r"]);
        assert_eq!(lex(r#""\x""#), Err(ParseErrorKind::BadEscape(Some('x'))));
        assert_eq!(lex(r#""abc\"#), Err(ParseErrorKind::BadEscape(None)));
        assert_eq!(lex(r#""abc"#), Err(ParseErrorKind::UnterminatedString));
        assert_eq!(lex(r#""a"b"#), Err(ParseErrorKind::UnexpectedChar('b')));
        assert_eq!(lex(r"a\b"), Err(ParseErrorKind::UnexpectedChar('\\')));
    }

    #[test]
    fn numbers_rejoin_only_adjacent_tokens() {
        assert_eq!(number("12"), Ok(12.0));
        assert_eq!(number("-12.5"), Ok(-12.5));
        assert_eq!(number(".5"), Ok(0.5));
        assert_eq!(number("1.5e-3"), Ok(1.5e-3));
        assert_eq!(number("-0"), Ok(-0.0));
        assert_eq!(number("1 . 5"), Err(ParseErrorKind::Expected("end")));
        assert_eq!(
            number("1.5.5"),
            Err(ParseErrorKind::BadNumber("1.5.5".into()))
        );
        assert_eq!(number("inf"), Err(ParseErrorKind::BadNumber("inf".into())));
        assert_eq!(number("NaN"), Err(ParseErrorKind::BadNumber("NaN".into())));
        assert_eq!(
            number("1e39"),
            Err(ParseErrorKind::BadNumber("1e39".into()))
        );
        assert_eq!(number("\"1\""), Err(ParseErrorKind::Expected("number")));
    }

    #[test]
    fn writer_quotes_exactly_when_needed() {
        let w = |s: &str| {
            let mut o = String::new();
            write_atom(&mut o, s);
            o
        };
        assert_eq!(w("clear_channel"), "clear_channel");
        assert_eq!(w("a/b:c@d-1+2"), "a/b:c@d-1+2");
        assert_eq!(w(""), "\"\"");
        assert_eq!(w("a b"), "\"a b\"");
        assert_eq!(w("1.5"), "\"1.5\"");
        assert_eq!(w("->"), "\"->\"");
        assert_eq!(w("é"), "\"é\"");
        assert_eq!(w("a\"\\\n"), r#""a\"\\\n""#);
    }

    #[test]
    fn every_written_atom_lexes_back() {
        for s in [
            "", "x", "a b", "#", "=", ".", "\"", "\\", "\n\r\t", "->", "node", "é ü",
        ] {
            let mut o = String::new();
            write_atom(&mut o, s);
            let toks: Vec<Tok> = lex(&o).unwrap().into_iter().map(|l| l.tok).collect();
            assert_eq!(
                toks,
                vec![Tok::Atom {
                    text: s.to_string(),
                    quoted: o.starts_with('"'),
                }],
                "atom {s:?} written as {o:?}"
            );
        }
    }
}
