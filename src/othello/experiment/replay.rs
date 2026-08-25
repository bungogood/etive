use std::collections::VecDeque;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::SelfPlaySample;
use crate::game::{Game, Outcome};
use crate::othello::{BitBoard, Board, Color};

const REPLAY_MAGIC: &[u8; 8] = b"ETRP0001";
const HEADER_SIZE: u64 = 16;
const RECORD_SIZE: u64 = 286;

pub(super) fn trim_replay(replay: &mut VecDeque<Vec<SelfPlaySample>>, maximum: usize) {
    let samples = replay
        .iter()
        .fold(0usize, |total, shard| total.saturating_add(shard.len()));
    let mut excess = samples.saturating_sub(maximum);
    while excess > 0 {
        let oldest = replay.front_mut().expect("excess implies non-empty replay");
        if excess >= oldest.len() {
            excess -= oldest.len();
            replay.pop_front();
        } else {
            oldest.drain(..excess);
            excess = 0;
        }
    }
}

pub(super) fn load_replay(
    output: &Path,
    generation: usize,
    maximum: usize,
) -> Result<VecDeque<Vec<SelfPlaySample>>, Box<dyn Error>> {
    let mut replay = VecDeque::new();
    let mut samples = 0usize;
    for generation in (1..=generation).rev() {
        let path = replay_path(output, generation);
        if !path.exists() {
            continue;
        }
        let shard = read_replay(&path)?;
        samples = samples.saturating_add(shard.len());
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
    for sample in samples {
        validate_sample(sample)?;
    }
    let count = u64::try_from(samples.len()).map_err(|_| invalid_replay("too many samples"))?;
    let temporary = temporary_replay_path(path);
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writer.write_all(REPLAY_MAGIC)?;
    writer.write_all(&count.to_le_bytes())?;
    for sample in samples {
        writer.write_all(&sample.position.discs(Color::Black).0.to_le_bytes())?;
        writer.write_all(&sample.position.discs(Color::White).0.to_le_bytes())?;
        writer.write_all(&[match sample.position.side_to_move() {
            Color::Black => 0,
            Color::White => 1,
        }])?;
        for probability in sample.policy {
            writer.write_all(&probability.to_le_bytes())?;
        }
        writer.write_all(&[match sample.outcome {
            Outcome::Loss => 0,
            Outcome::Draw => 1,
            Outcome::Win => 2,
        }])?;
        writer.write_all(&sample.game.to_le_bytes())?;
    }
    writer.flush()?;
    drop(writer);
    fs::rename(temporary, path)?;
    Ok(())
}

pub(super) fn read_replay(path: &Path) -> Result<Vec<SelfPlaySample>, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut magic = [0; 8];
    file.read_exact(&mut magic)?;
    if &magic != REPLAY_MAGIC {
        return Err(invalid_replay("invalid replay header"));
    }
    let count = read_u64(&mut file)?;
    let expected_size = count
        .checked_mul(RECORD_SIZE)
        .and_then(|size| HEADER_SIZE.checked_add(size))
        .ok_or_else(|| invalid_replay("replay size overflow"))?;
    if file_size != expected_size {
        return Err(invalid_replay("replay file size does not match header"));
    }
    let count = usize::try_from(count).map_err(|_| invalid_replay("replay count is too large"))?;
    let mut reader = BufReader::new(file);
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(count)
        .map_err(|_| invalid_replay("replay count is too large"))?;
    for _ in 0..count {
        let black = read_u64(&mut reader)?;
        let white = read_u64(&mut reader)?;
        let side = match read_u8(&mut reader)? {
            0 => Color::Black,
            1 => Color::White,
            _ => return Err(invalid_replay("invalid side to move")),
        };
        let position = Board::from_discs(BitBoard(black), BitBoard(white), side)
            .map_err(|_| invalid_replay("black and white discs overlap"))?;
        let mut policy = [0.0; 65];
        for probability in &mut policy {
            *probability = read_f32(&mut reader)?;
        }
        let outcome = match read_u8(&mut reader)? {
            0 => Outcome::Loss,
            1 => Outcome::Draw,
            2 => Outcome::Win,
            _ => return Err(invalid_replay("invalid outcome")),
        };
        let sample = SelfPlaySample {
            position,
            policy,
            outcome,
            game: read_u64(&mut reader)?,
        };
        validate_sample(&sample)?;
        samples.push(sample);
    }
    Ok(samples)
}

fn validate_sample(sample: &SelfPlaySample) -> Result<(), Box<dyn Error>> {
    let mut sum = 0.0f64;
    for (index, &probability) in sample.policy.iter().enumerate() {
        if !probability.is_finite() || probability < 0.0 {
            return Err(invalid_replay(
                "policy probabilities must be finite and nonnegative",
            ));
        }
        if probability > 0.0 {
            let action = Board::action_from_index(index)
                .ok_or_else(|| invalid_replay("invalid policy action"))?;
            if !sample.position.is_legal(action) {
                return Err(invalid_replay(
                    "policy assigns probability to illegal action",
                ));
            }
        }
        sum += f64::from(probability);
    }
    if (sum - 1.0).abs() > 1e-5 {
        return Err(invalid_replay("policy probabilities must sum to one"));
    }
    Ok(())
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut bytes = [0];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn invalid_replay(message: &str) -> Box<dyn Error> {
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

fn temporary_replay_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "etive-replay-{}-{}-{name}.bin",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        ))
    }

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

    fn assert_invalid(result: Result<Vec<SelfPlaySample>, Box<dyn Error>>) {
        let error = result.unwrap_err();
        assert_eq!(
            error.downcast_ref::<io::Error>().map(io::Error::kind),
            Some(io::ErrorKind::InvalidData)
        );
    }

    #[test]
    fn replay_round_trips() {
        let path = path("round-trip");
        let samples = [sample(42)];

        atomic_replay_save(&samples, &path).unwrap();
        let loaded = read_replay(&path).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].position, samples[0].position);
        assert_eq!(loaded[0].policy, samples[0].policy);
        assert_eq!(loaded[0].outcome, Outcome::Win);
        assert_eq!(loaded[0].game, 42);
    }

    #[test]
    fn rejects_trailing_data() {
        let path = path("trailing");
        atomic_replay_save(&[sample(1)], &path).unwrap();
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0])
            .unwrap();

        assert_invalid(read_replay(&path));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_malformed_policies_on_save() {
        let path = path("malformed");
        for policy in [
            {
                let mut policy = sample(1).policy;
                policy[19] = f32::NAN;
                policy
            },
            {
                let mut policy = sample(1).policy;
                policy[19] = -1.0;
                policy[26] = 2.0;
                policy
            },
            {
                let mut policy = sample(1).policy;
                policy[19] = 0.5;
                policy
            },
            {
                let mut policy = sample(1).policy;
                policy[19] = 0.0;
                policy[0] = 1.0;
                policy
            },
        ] {
            let mut invalid = sample(1);
            invalid.policy = policy;
            assert!(atomic_replay_save(&[invalid], &path).is_err());
        }
        assert!(!path.exists());
    }

    #[test]
    fn rejects_malformed_policy_on_load() {
        let path = path("malformed-load");
        atomic_replay_save(&[sample(1)], &path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes[33..37].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        assert_invalid(read_replay(&path));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn trims_to_exact_capacity_with_oversized_shards() {
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
