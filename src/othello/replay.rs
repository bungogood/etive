use std::collections::{HashMap, VecDeque, hash_map::Entry};
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::game::{Game, Outcome};
use crate::othello::Board;

use super::temporary::atomic_file_save;

pub type SelfPlaySample = crate::self_play::Sample<Board>;

const FORMAT_VERSION: u8 = 2;

pub(super) fn trim_replay(replay: &mut VecDeque<Vec<SelfPlaySample>>, maximum: usize) {
    let total = replay.iter().map(Vec::len).sum::<usize>();
    let mut excess = total.saturating_sub(maximum);
    while excess > 0 {
        let oldest = replay.front_mut().expect("excess implies replay data");
        if excess < oldest.len() {
            oldest.drain(..excess);
            break;
        }
        excess -= oldest.len();
        replay.pop_front();
    }
}

pub(super) fn load_replay(
    output: &Path,
    generation: usize,
    maximum: usize,
) -> Result<VecDeque<Vec<SelfPlaySample>>, Box<dyn Error>> {
    let mut replay = VecDeque::new();
    let mut samples = 0;
    for generation in (1..=generation).rev() {
        let path = replay_path(output, generation);
        if !path.exists() {
            return Err(format!("missing replay shard: {}", path.display()).into());
        }
        let shard = read_replay(&path)?;
        samples += shard.len();
        replay.push_front(shard);
        if samples >= maximum {
            break;
        }
    }
    trim_replay(&mut replay, maximum);
    Ok(replay)
}

pub(super) fn atomic_replay_save(
    samples: &[SelfPlaySample],
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    validate_samples(samples)?;
    let bytes = bincode::encode_to_vec((FORMAT_VERSION, samples), bincode::config::standard())?;
    atomic_file_save(path, |temporary| {
        fs::write(temporary, bytes)?;
        Ok(())
    })
}

pub(crate) fn read_replay(path: &Path) -> Result<Vec<SelfPlaySample>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let ((version, samples), consumed): ((u8, Vec<SelfPlaySample>), usize) =
        bincode::decode_from_slice(&bytes, bincode::config::standard())?;
    if version != FORMAT_VERSION || consumed != bytes.len() {
        return Err(invalid("unsupported or malformed replay"));
    }
    validate_samples(&samples)?;
    Ok(samples)
}

fn validate_samples(samples: &[SelfPlaySample]) -> Result<(), Box<dyn Error>> {
    let mut game_results = HashMap::new();
    for sample in samples {
        validate(sample)?;
        let winner = match sample.outcome {
            Outcome::Win => Some(sample.position.side_to_move()),
            Outcome::Draw => None,
            Outcome::Loss => Some(!sample.position.side_to_move()),
        };
        match game_results.entry(sample.game) {
            Entry::Vacant(entry) => {
                entry.insert(winner);
            }
            Entry::Occupied(entry) if *entry.get() != winner => {
                return Err(invalid("samples from one game disagree on the outcome"));
            }
            Entry::Occupied(_) => {}
        }
    }
    Ok(())
}

fn validate(sample: &SelfPlaySample) -> Result<(), Box<dyn Error>> {
    let mut sum = 0.0f64;
    for (index, &probability) in sample.policy.iter().enumerate() {
        if !probability.is_finite() || probability < 0.0 {
            return Err(invalid("policy must contain finite probabilities"));
        }
        if probability > 0.0 {
            let action =
                Board::action_from_index(index).ok_or_else(|| invalid("invalid action"))?;
            if !sample.position.is_legal(action) {
                return Err(invalid("policy assigns probability to an illegal action"));
            }
        }
        sum += f64::from(probability);
    }
    if (sum - 1.0).abs() > 1e-5 {
        return Err(invalid("policy probabilities must sum to one"));
    }
    Ok(())
}

fn invalid(message: &str) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidData, message).into()
}

pub(super) fn replay_path(output: &Path, generation: usize) -> PathBuf {
    output
        .join("replay")
        .join(format!("generation-{generation:04}.bin"))
}

pub(super) fn validation_replay_path(output: &Path, generation: usize) -> PathBuf {
    output
        .join("replay")
        .join(format!("generation-{generation:04}-validation.bin"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::othello::Move;

    fn sample(game: u64) -> SelfPlaySample {
        let mut policy = [0.0; 65];
        policy[19] = 1.0;
        SelfPlaySample {
            position: Board::default(),
            policy,
            outcome: Outcome::Win,
            game,
        }
    }

    fn sample_for(position: Board, outcome: Outcome, game: u64) -> SelfPlaySample {
        let mut policy = [0.0; Board::ACTION_COUNT];
        let action = position.legal_actions().next().unwrap();
        policy[Board::action_index(action)] = 1.0;
        SelfPlaySample {
            position,
            policy,
            outcome,
            game,
        }
    }

    #[test]
    fn replay_round_trips_and_rejects_trailing_data() {
        let path = std::env::temp_dir().join(format!("etive-replay-{}.bin", std::process::id()));
        atomic_replay_save(&[sample(42)], &path).unwrap();
        assert_eq!(read_replay(&path).unwrap()[0].game, 42);

        let mut bytes = fs::read(&path).unwrap();
        bytes.push(0);
        fs::write(&path, bytes).unwrap();
        assert!(read_replay(&path).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_invalid_policies() {
        let path = std::env::temp_dir().join(format!("etive-invalid-{}.bin", std::process::id()));
        for (index, value) in [(19, f32::NAN), (19, -1.0), (0, 1.0)] {
            let mut sample = sample(1);
            sample.policy[19] = 0.0;
            sample.policy[index] = value;
            assert!(atomic_replay_save(&[sample], &path).is_err());
        }
    }

    #[test]
    fn rejects_inconsistent_value_perspectives_within_a_game() {
        let path = std::env::temp_dir().join(format!(
            "etive-inconsistent-outcome-{}.bin",
            std::process::id()
        ));
        let black_position = Board::default();
        let mut white_position = black_position;
        white_position.play(Move::Place("d3".parse().unwrap()));

        let consistent = [
            sample_for(black_position, Outcome::Win, 42),
            sample_for(white_position, Outcome::Loss, 42),
        ];
        atomic_replay_save(&consistent, &path).unwrap();

        let inconsistent = [
            sample_for(black_position, Outcome::Win, 42),
            sample_for(white_position, Outcome::Win, 42),
        ];
        assert!(atomic_replay_save(&inconsistent, &path).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn trimming_keeps_the_newest_exact_capacity() {
        let mut replay = VecDeque::from([
            (0..3).map(sample).collect::<Vec<_>>(),
            (3..9).map(sample).collect::<Vec<_>>(),
        ]);
        trim_replay(&mut replay, 4);
        assert_eq!(
            replay
                .iter()
                .flatten()
                .map(|sample| sample.game)
                .collect::<Vec<_>>(),
            vec![5, 6, 7, 8]
        );
    }
}
