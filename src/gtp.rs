use std::io::{self, BufRead, Write};

use etive::evaluator::OthelloCandleEvaluator;
use etive::mcts::{Mcts, MctsConfig, SearchWorkspace};
use etive::othello::{Board, Color, GameStatus, Move};

macro_rules! commands {
    ($($variant:ident = $name:literal),+ $(,)?) => {
        #[derive(Clone, Copy)]
        enum GtpCommand {
            $($variant),+
        }

        const COMMANDS: &[(&str, GtpCommand)] = &[
            $(($name, GtpCommand::$variant)),+
        ];
    };
}

commands! {
    ProtocolVersion = "protocol_version",
    Name = "name",
    Version = "version",
    KnownCommand = "known_command",
    ListCommands = "list_commands",
    Quit = "quit",
    BoardSize = "boardsize",
    ClearBoard = "clear_board",
    Komi = "komi",
    Play = "play",
    GenMove = "genmove",
    RegGenMove = "reg_genmove",
    Undo = "undo",
    SetGame = "set_game",
    ListGames = "list_games",
    ShowBoard = "showboard",
    FinalScore = "final_score",
}

pub(crate) fn run(reader: impl BufRead, mut writer: impl Write) -> io::Result<()> {
    run_session(reader, &mut writer, Session::default())
}

pub(crate) fn run_with_evaluator(
    reader: impl BufRead,
    mut writer: impl Write,
    evaluator: OthelloCandleEvaluator,
    simulations: u32,
    batch_size: usize,
) -> io::Result<()> {
    run_session(
        reader,
        &mut writer,
        Session {
            board: Board::default(),
            history: Vec::new(),
            search: Some(SearchEngine {
                evaluator,
                simulations,
                workspace: SearchWorkspace::new(batch_size),
                tree: None,
            }),
        },
    )
}

fn run_session(
    reader: impl BufRead,
    writer: &mut impl Write,
    mut session: Session,
) -> io::Result<()> {
    for line in reader.lines() {
        let Some(response) = session.execute(&line?) else {
            continue;
        };
        writer.write_all(response.render().as_bytes())?;
        writer.flush()?;
        if response.quit {
            break;
        }
    }
    Ok(())
}

#[derive(Default)]
struct Session {
    board: Board,
    history: Vec<Board>,
    search: Option<SearchEngine>,
}

struct SearchEngine {
    evaluator: OthelloCandleEvaluator,
    simulations: u32,
    workspace: SearchWorkspace<Board>,
    tree: Option<Mcts<Board>>,
}

impl SearchEngine {
    fn best_move(&mut self, board: Board) -> Result<Move, String> {
        if self
            .tree
            .as_ref()
            .is_some_and(|tree| tree.root_position() != &board)
        {
            self.tree = None;
        }
        let tree = self
            .tree
            .get_or_insert_with(|| Mcts::new(board, MctsConfig::default()));
        self.workspace
            .run_parallel(
                std::slice::from_mut(tree),
                &mut self.evaluator,
                self.simulations,
            )
            .map_err(|error| error.to_string())?;
        tree.best_action()
            .ok_or_else(|| "search found no legal action".to_owned())
    }

    fn advance(&mut self, mv: Move) {
        if !self.tree.as_mut().is_some_and(|tree| tree.advance(mv)) {
            self.tree = None;
        }
    }

    fn reset(&mut self) {
        self.tree = None;
    }
}

impl Session {
    fn execute(&mut self, line: &str) -> Option<Response> {
        let input = line.split('#').next().unwrap_or_default();
        let mut fields = input.split_ascii_whitespace().peekable();
        let first = fields.next()?;
        let id = first
            .bytes()
            .all(|byte| byte.is_ascii_digit())
            .then(|| first.to_owned());
        let name = if id.is_some() {
            match fields.next() {
                Some(command) => command,
                None => return Some(Response::failure(id, "missing command")),
            }
        } else {
            first
        };
        let arguments: Vec<_> = fields.collect();
        let Some(command) = parse_command(name) else {
            return Some(Response::failure(id, "unknown command"));
        };

        let result = match command {
            GtpCommand::ProtocolVersion => no_arguments(&arguments).map(|()| "2".to_owned()),
            GtpCommand::Name => no_arguments(&arguments).map(|()| "Etive".to_owned()),
            GtpCommand::Version => {
                no_arguments(&arguments).map(|()| env!("CARGO_PKG_VERSION").to_owned())
            }
            GtpCommand::KnownCommand => {
                one_argument(&arguments).map(|name| parse_command(name).is_some().to_string())
            }
            GtpCommand::ListCommands => no_arguments(&arguments).map(|()| command_names()),
            GtpCommand::BoardSize => self.boardsize(&arguments),
            GtpCommand::ClearBoard => no_arguments(&arguments).map(|()| self.clear()),
            GtpCommand::Komi => parse_komi(&arguments),
            GtpCommand::Play => self.play(&arguments),
            GtpCommand::GenMove => self.genmove(&arguments, true),
            GtpCommand::RegGenMove => self.genmove(&arguments, false),
            GtpCommand::Undo => self.undo(&arguments),
            GtpCommand::SetGame => set_game(&arguments),
            GtpCommand::ListGames => no_arguments(&arguments).map(|()| "Othello".to_owned()),
            GtpCommand::ShowBoard => no_arguments(&arguments).map(|()| self.board.to_string()),
            GtpCommand::FinalScore => no_arguments(&arguments).map(|()| self.final_score()),
            GtpCommand::Quit => {
                return Some(match no_arguments(&arguments) {
                    Ok(()) => Response::success(id, String::new()).with_quit(),
                    Err(error) => Response::failure(id, error),
                });
            }
        };

        Some(match result {
            Ok(body) => Response::success(id, body),
            Err(error) => Response::failure(id, error),
        })
    }

    fn boardsize(&mut self, arguments: &[&str]) -> Result<String, String> {
        let size = one_argument(arguments)?;
        if size != "8" {
            return Err("unacceptable size".to_owned());
        }
        Ok(self.clear())
    }

    fn clear(&mut self) -> String {
        self.board = Board::default();
        self.history.clear();
        if let Some(search) = &mut self.search {
            search.reset();
        }
        String::new()
    }

    fn play(&mut self, arguments: &[&str]) -> Result<String, String> {
        if arguments.len() != 2 {
            return Err("expected color and move".to_owned());
        }
        self.require_turn(arguments[0])?;
        let mv = arguments[1]
            .parse::<Move>()
            .map_err(|_| "invalid move".to_owned())?;
        self.apply_move(mv)?;
        Ok(String::new())
    }

    fn genmove(&mut self, arguments: &[&str], play_move: bool) -> Result<String, String> {
        let color = one_argument(arguments)?;
        self.require_turn(color)?;
        if self.board.status() != GameStatus::Ongoing {
            return Ok("pass".to_owned());
        }

        let mv = match &mut self.search {
            Some(search) => search.best_move(self.board)?,
            None => match self.board.legal_placements().into_iter().next() {
                Some(square) => Move::Place(square),
                None if self.board.is_pass_legal() => Move::Pass,
                None => return Ok("pass".to_owned()),
            },
        };
        if play_move {
            self.apply_move(mv)?;
        }
        Ok(mv.to_string())
    }

    fn apply_move(&mut self, mv: Move) -> Result<(), String> {
        let previous = self.board;
        self.board
            .try_play(mv)
            .map_err(|_| "illegal move".to_owned())?;
        self.history.push(previous);
        if let Some(search) = &mut self.search {
            search.advance(mv);
        }
        Ok(())
    }

    fn require_turn(&self, color: &str) -> Result<(), String> {
        let color = parse_color(color)?;
        if color != self.board.side_to_move() {
            return Err("wrong color".to_owned());
        }
        Ok(())
    }

    fn undo(&mut self, arguments: &[&str]) -> Result<String, String> {
        no_arguments(arguments)?;
        self.board = self.history.pop().ok_or_else(|| "cannot undo".to_owned())?;
        if let Some(search) = &mut self.search {
            search.reset();
        }
        Ok(String::new())
    }

    fn final_score(&self) -> String {
        let black = self.board.discs(Color::Black).len();
        let white = self.board.discs(Color::White).len();
        match self.board.status() {
            GameStatus::Won(Color::Black) => format!("B+{}", black - white),
            GameStatus::Won(Color::White) => format!("W+{}", white - black),
            GameStatus::Drawn => "0".to_owned(),
            GameStatus::Ongoing => "unknown".to_owned(),
        }
    }
}

fn parse_command(name: &str) -> Option<GtpCommand> {
    COMMANDS
        .iter()
        .find_map(|&(candidate, command)| (candidate == name).then_some(command))
}

fn command_names() -> String {
    COMMANDS
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join("\n")
}

struct Response {
    id: Option<String>,
    body: String,
    success: bool,
    quit: bool,
}

impl Response {
    fn success(id: Option<String>, body: String) -> Self {
        Self {
            id,
            body,
            success: true,
            quit: false,
        }
    }

    fn failure(id: Option<String>, body: impl Into<String>) -> Self {
        Self {
            id,
            body: body.into(),
            success: false,
            quit: false,
        }
    }

    fn with_quit(mut self) -> Self {
        self.quit = true;
        self
    }

    fn render(&self) -> String {
        let marker = if self.success { '=' } else { '?' };
        let id = self.id.as_deref().unwrap_or_default();
        if self.body.is_empty() {
            format!("{marker}{id}\n\n")
        } else {
            format!("{marker}{id} {}\n\n", self.body)
        }
    }
}

fn no_arguments(arguments: &[&str]) -> Result<(), String> {
    if arguments.is_empty() {
        Ok(())
    } else {
        Err("unexpected arguments".to_owned())
    }
}

fn one_argument<'a>(arguments: &'a [&str]) -> Result<&'a str, String> {
    match arguments {
        [argument] => Ok(argument),
        _ => Err("expected one argument".to_owned()),
    }
}

fn parse_color(color: &str) -> Result<Color, String> {
    if color.eq_ignore_ascii_case("black") || color.eq_ignore_ascii_case("b") {
        Ok(Color::Black)
    } else if color.eq_ignore_ascii_case("white") || color.eq_ignore_ascii_case("w") {
        Ok(Color::White)
    } else {
        Err("invalid color".to_owned())
    }
}

fn parse_komi(arguments: &[&str]) -> Result<String, String> {
    let komi = one_argument(arguments)?
        .parse::<f64>()
        .map_err(|_| "invalid komi".to_owned())?;
    if komi.is_finite() && komi == 0.0 {
        Ok(String::new())
    } else {
        Err("Othello does not support komi".to_owned())
    }
}

fn set_game(arguments: &[&str]) -> Result<String, String> {
    let game = one_argument(arguments)?;
    if game.eq_ignore_ascii_case("othello") {
        Ok(String::new())
    } else {
        Err("unsupported game".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::str::FromStr;

    use candle_core::Device;

    use super::*;
    use etive::othello::{BitBoard, Square};

    fn exchange(input: &str) -> String {
        let mut output = Vec::new();
        run(Cursor::new(input), &mut output).unwrap();
        String::from_utf8(output).unwrap()
    }

    #[test]
    fn reports_protocol_metadata_with_command_ids() {
        assert_eq!(
            exchange("1 protocol_version\n2 name\n3 version\n4 quit\n"),
            format!(
                "=1 2\n\n=2 Etive\n\n=3 {}\n\n=4\n\n",
                env!("CARGO_PKG_VERSION")
            )
        );
    }

    #[test]
    fn rejects_unknown_commands_and_bad_arguments() {
        assert_eq!(
            exchange("9 boardsize 6\nwat\nkomi nope\nkomi 6.5\nquit now\n"),
            "?9 unacceptable size\n\n? unknown command\n\n? invalid komi\n\n? Othello does not support komi\n\n? unexpected arguments\n\n"
        );
    }

    #[test]
    fn plays_and_generates_moves_without_mutating_on_failure() {
        assert_eq!(
            exchange("play black a1\ngenmove black\nplay black c3\ngenmove white\nquit\n"),
            "? illegal move\n\n= d3\n\n? wrong color\n\n= c3\n\n=\n\n"
        );
    }

    #[test]
    fn regression_generation_and_undo_preserve_expected_state() {
        assert_eq!(
            exchange("reg_genmove b\nreg_genmove black\ngenmove b\nundo\ngenmove black\nquit\n"),
            "= d3\n\n= d3\n\n= d3\n\n=\n\n= d3\n\n=\n\n"
        );
    }

    #[test]
    fn handles_forced_passes() {
        let a1 = Square::from_str("a1").unwrap().bitboard();
        let b1 = Square::from_str("b1").unwrap().bitboard();
        let mut session = Session {
            board: Board::from_discs(a1, b1, Color::White).unwrap(),
            history: Vec::new(),
            search: None,
        };

        let response = session.execute("genmove white").unwrap();
        assert_eq!(response.render(), "= pass\n\n");
        assert_eq!(session.board.side_to_move(), Color::Black);
        assert_eq!(
            session.board.legal_placements(),
            Square::from_str("c1").unwrap().bitboard()
        );
    }

    #[test]
    fn reports_terminal_scores() {
        let mut session = Session {
            board: Board::from_discs(BitBoard::FULL, BitBoard::EMPTY, Color::White).unwrap(),
            history: Vec::new(),
            search: None,
        };
        assert_eq!(
            session.execute("final_score").unwrap().render(),
            "= B+64\n\n"
        );
    }

    #[test]
    fn checkpoint_search_generates_a_legal_move() {
        let evaluator = OthelloCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let mut session = Session {
            board: Board::default(),
            history: Vec::new(),
            search: Some(SearchEngine {
                evaluator,
                simulations: 8,
                workspace: SearchWorkspace::new(8),
                tree: None,
            }),
        };

        let response = session.execute("genmove black").unwrap();
        let mv = response.body.parse::<Move>().unwrap();

        assert!(Board::default().is_legal(mv));
        assert_eq!(session.history, vec![Board::default()]);
        assert_eq!(
            session
                .search
                .as_ref()
                .unwrap()
                .tree
                .as_ref()
                .unwrap()
                .root_position(),
            &session.board
        );
        let search = session.search.as_ref().unwrap();
        assert!(search.evaluator.batches() < search.evaluator.evaluations());
    }

    #[test]
    fn checkpoint_search_advances_for_opponent_moves_and_resets_on_undo() {
        let evaluator = OthelloCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let mut session = Session {
            board: Board::default(),
            history: Vec::new(),
            search: Some(SearchEngine {
                evaluator,
                simulations: 2,
                workspace: SearchWorkspace::new(2),
                tree: None,
            }),
        };

        session.execute("genmove black").unwrap();
        let opponent_move = session
            .search
            .as_ref()
            .unwrap()
            .tree
            .as_ref()
            .unwrap()
            .best_action()
            .unwrap();
        session
            .execute(&format!("play white {opponent_move}"))
            .unwrap();

        assert_eq!(
            session
                .search
                .as_ref()
                .unwrap()
                .tree
                .as_ref()
                .unwrap()
                .root_position(),
            &session.board
        );

        session.execute("undo").unwrap();
        assert!(session.search.as_ref().unwrap().tree.is_none());
    }

    #[test]
    fn checkpoint_search_handles_terminal_positions_without_inference() {
        let evaluator = OthelloCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let mut session = Session {
            board: Board::from_discs(BitBoard::FULL, BitBoard::EMPTY, Color::White).unwrap(),
            history: Vec::new(),
            search: Some(SearchEngine {
                evaluator,
                simulations: 8,
                workspace: SearchWorkspace::new(8),
                tree: None,
            }),
        };

        assert_eq!(
            session.execute("genmove white").unwrap().render(),
            "= pass\n\n"
        );
        assert_eq!(session.search.as_ref().unwrap().evaluator.evaluations(), 0);
    }
}
