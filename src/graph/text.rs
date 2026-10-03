//! A dependency-free, line-oriented text format for [`Graph`].
//!
//! Graphs need to be saved, diffed, hand-edited and loaded by tools that don't
//! link this crate (shell scripts, the example runner, a future editor). A
//! tiny line format serves all of that without pulling in serde: one
//! statement per line, so a diff of two graph files reads as a diff of nodes
//! and edges.
//!
//! ```text
//! # pipe-graph text format v1
//! node split split channels=3
//! node merge merge channels=3
//! node "my node" crop h=2 w=2 x=0 y=0
//! edge split.out0 -> merge.in0
//! edge "my node".out -> split.in
//! ```
//!
//! # Grammar
//!
//! ```text
//! file      = { line "\n" } [ line ]       ; a trailing "\r" on a line is ignored
//! line      = ws* [ statement ] ws* [ comment ]
//! comment   = "#" { any char }             ; only outside a quoted atom
//! statement = node | edge
//! node      = "node" atom atom { param }   ; id, kind, params
//! param     = atom "=" atom                ; key=value
//! edge      = "edge" endpoint "->" endpoint
//! endpoint  = atom "." atom                ; node.port
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
//! * `node`, `edge` and `->` must be written bare: a quoted `"->"` is a
//!   port/id named `->`, not the arrow. Inside quotes every character except
//!   `"` and `\` is literal, so ids, kinds, ports, keys and values may contain
//!   spaces, `=`, `.`, `#` or non-ASCII text.
//! * A quoted atom must be followed by whitespace, `=`, `.`, a comment or the
//!   end of the line: `"a"b` is rejected rather than silently read as two atoms.
//! * Statements may appear in any order; an edge may name a node declared on a
//!   later line. Duplicate node ids, duplicate param keys on one node, edges
//!   to undeclared nodes and malformed lines are reported as a [`ParseError`]
//!   carrying the 1-based line number.
//!
//! # Determinism
//!
//! [`Graph::to_text`] writes a header comment, then nodes sorted by id (params
//! sorted by key), then edges in ascending [`EdgeId`] order — i.e. the order
//! they were connected. It emits an atom bare only when it is non-empty and
//! consists solely of ASCII alphanumerics and `_ - + / : @`; everything else is
//! quoted. The output is therefore a pure function of the graph's structure,
//! and `from_text(to_text(g)).to_text() == to_text(g)`.
//!
//! Edge ids are **not** serialized: [`Graph::from_text`] re-connects edges in
//! file order, so a re-loaded graph gets fresh ids `0..n` in the same relative
//! order as the original.

use std::fmt;

use super::{Connection, EdgeId, Graph, GraphError, NodeId, NodeSpec, Params, PortId};

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
    /// The first word of a statement was neither `node` nor `edge`.
    UnknownStatement(String),
    /// The statement ended or continued with something other than `expected`.
    Expected(&'static str),
    /// Two `node` statements declared the same id.
    DuplicateNodeId(String),
    /// One `node` statement repeated a param key.
    DuplicateParam(String),
    /// An `edge` names a node that no `node` statement declares.
    UnknownNode(String),
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
                write!(f, "unknown statement '{s}' (expected 'node' or 'edge')")
            }
            ParseErrorKind::Expected(what) => write!(f, "expected {what}"),
            ParseErrorKind::DuplicateNodeId(id) => write!(f, "duplicate node id '{id}'"),
            ParseErrorKind::DuplicateParam(k) => write!(f, "duplicate parameter '{k}'"),
            ParseErrorKind::UnknownNode(id) => write!(f, "edge references unknown node '{id}'"),
        }
    }
}

impl std::error::Error for ParseError {}

impl Graph {
    /// Serialize to the line-oriented text format (see the [module docs](self)).
    ///
    /// Output is deterministic: nodes by id, params by key, edges by
    /// [`EdgeId`].
    pub fn to_text(&self) -> String {
        let mut out = String::from("# pipe-graph text format v1\n");

        let mut nodes: Vec<&NodeSpec> = self.nodes.values().collect();
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

        for (_, conn) in self.edges_in_order() {
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
        out
    }

    /// Parse the text format (see the [module docs](self)) into a new graph.
    ///
    /// Edges are connected in file order and receive fresh [`EdgeId`]s.
    pub fn from_text(text: &str) -> Result<Graph, ParseError> {
        let mut graph = Graph::new();
        // Edges are resolved after all nodes are known, so files may declare
        // nodes after the edges that use them.
        let mut edges: Vec<(usize, Endpoint, Endpoint)> = Vec::new();

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
            match head {
                Tok::Atom {
                    text,
                    quoted: false,
                } if text == "node" => {
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
                Tok::Atom {
                    text,
                    quoted: false,
                } if text == "edge" => {
                    let from = p.endpoint().map_err(err)?;
                    match p.next() {
                        Some(Tok::Atom {
                            text,
                            quoted: false,
                        }) if text == "->" => {}
                        _ => return Err(err(ParseErrorKind::Expected("'->' between endpoints"))),
                    }
                    let to = p.endpoint().map_err(err)?;
                    if p.peek().is_some() {
                        return Err(err(ParseErrorKind::Expected("end of line after edge")));
                    }
                    edges.push((line, from, to));
                }
                Tok::Atom { text, .. } => {
                    return Err(err(ParseErrorKind::UnknownStatement(text.clone())));
                }
                _ => return Err(err(ParseErrorKind::Expected("'node' or 'edge'"))),
            }
        }

        for (line, from, to) in edges {
            graph.connect(from, to).map_err(|e| ParseError {
                line,
                kind: match e {
                    GraphError::MissingNode(id) => ParseErrorKind::UnknownNode(id),
                    _ => unreachable!("connect returned {e:?}"),
                },
            })?;
        }
        Ok(graph)
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

/// Characters that end a bare atom.
fn is_delim(c: char) -> bool {
    c == ' ' || c == '\t' || matches!(c, '"' | '#' | '=' | '.' | '\\')
}

fn lex(line: &str) -> Result<Vec<Tok>, ParseErrorKind> {
    let mut toks = Vec::new();
    let mut chars = line.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' => {
                chars.next();
            }
            '#' => break,
            '=' => {
                chars.next();
                toks.push(Tok::Eq);
            }
            '.' => {
                chars.next();
                toks.push(Tok::Dot);
            }
            '\\' => return Err(ParseErrorKind::UnexpectedChar('\\')),
            '"' => {
                chars.next();
                let mut text = String::new();
                loop {
                    match chars.next() {
                        None => return Err(ParseErrorKind::UnterminatedString),
                        Some('"') => break,
                        Some('\\') => match chars.next() {
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
                if let Some(&next) = chars.peek()
                    && !matches!(next, ' ' | '\t' | '=' | '.' | '#')
                {
                    return Err(ParseErrorKind::UnexpectedChar(next));
                }
                toks.push(Tok::Atom { text, quoted: true });
            }
            _ => {
                let mut text = String::new();
                while let Some(&c) = chars.peek() {
                    if is_delim(c) {
                        break;
                    }
                    text.push(c);
                    chars.next();
                }
                toks.push(Tok::Atom {
                    text,
                    quoted: false,
                });
            }
        }
    }
    Ok(toks)
}

struct Cursor<'a> {
    toks: &'a [Tok],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn peek(&self) -> Option<&'a Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<&'a Tok> {
        let t = self.toks.get(self.pos);
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

    fn endpoint(&mut self) -> Result<Endpoint, ParseErrorKind> {
        let node = self.atom("node id in edge endpoint")?;
        self.expect(&Tok::Dot, "'.' between node and port")?;
        let port = self.atom("port name in edge endpoint")?;
        Ok((NodeId(node), PortId(port)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atoms(line: &str) -> Vec<String> {
        lex(line)
            .unwrap()
            .into_iter()
            .map(|t| match t {
                Tok::Atom { text, .. } => text,
                Tok::Eq => "=".into(),
                Tok::Dot => ".".into(),
            })
            .collect()
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
    fn lexes_escapes() {
        assert_eq!(atoms(r#""q\"\\\n\t\r""#), ["q\"\\\n\t\r"]);
        assert_eq!(lex(r#""\x""#), Err(ParseErrorKind::BadEscape(Some('x'))));
        assert_eq!(lex(r#""abc\"#), Err(ParseErrorKind::BadEscape(None)));
        assert_eq!(lex(r#""abc"#), Err(ParseErrorKind::UnterminatedString));
        assert_eq!(lex(r#""a"b"#), Err(ParseErrorKind::UnexpectedChar('b')));
        assert_eq!(lex(r"a\b"), Err(ParseErrorKind::UnexpectedChar('\\')));
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
            assert_eq!(
                lex(&o).unwrap(),
                vec![Tok::Atom {
                    text: s.to_string(),
                    quoted: o.starts_with('"'),
                }],
                "atom {s:?} written as {o:?}"
            );
        }
    }
}
