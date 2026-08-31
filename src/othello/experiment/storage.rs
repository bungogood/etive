use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use fs2::FileExt;
use serde::{Deserialize, Serialize};

use super::super::OthelloNetwork;
use super::super::replay::{SelfPlaySample, read_replay, validation_replay_path};
use super::super::temporary::atomic_file_save;
use super::super::training::TrainingSession;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RunState {
    pub(super) generation: usize,
    pub(super) champion_generation: usize,
    pub(super) elapsed_seconds: f64,
}

pub(super) const RUN_MARKER: &str = "etive-run-v1\n";

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GenerationMetrics {
    pub(super) generation: usize,
    pub(super) samples: usize,
    pub(super) training_samples: usize,
    pub(super) validation_samples: usize,
    pub(super) replay_samples: usize,
    pub(super) self_play_seconds: f64,
    pub(super) self_play_evaluations: u64,
    pub(super) self_play_evaluations_per_second: f64,
    pub(super) learning_rate: f64,
    pub(super) training_steps: usize,
    pub(super) training_seconds: f64,
    pub(super) policy_loss: f64,
    pub(super) policy_target_entropy: f64,
    pub(super) policy_kl: f64,
    pub(super) value_loss: f64,
    pub(super) validation_policy_loss: f64,
    pub(super) validation_policy_target_entropy: f64,
    pub(super) validation_policy_kl: f64,
    pub(super) validation_value_loss: f64,
    pub(super) evaluated: bool,
    pub(super) candidate_wins: usize,
    pub(super) baseline_wins: usize,
    pub(super) draws: usize,
    pub(super) pair_0: usize,
    pub(super) pair_0_5: usize,
    pub(super) pair_1: usize,
    pub(super) pair_1_5: usize,
    pub(super) pair_2: usize,
    pub(super) score: f64,
    pub(super) los: f64,
    pub(super) promoted: bool,
    pub(super) baseline_generation: usize,
    pub(super) champion_generation: usize,
    pub(super) checkpoint: PathBuf,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PendingSelfPlay {
    pub(super) generation: usize,
    pub(super) training_samples: usize,
    pub(super) validation_samples: usize,
    pub(super) elapsed_seconds: f64,
    pub(super) evaluations: u64,
    pub(super) unique_games: usize,
}

pub(super) struct GenerationSelfPlay {
    pub(super) training: Vec<SelfPlaySample>,
    pub(super) validation: Vec<SelfPlaySample>,
    pub(super) pending: PendingSelfPlay,
    pub(super) recovered: bool,
}

pub(super) struct RunLock {
    _file: File,
}

pub(super) fn discard_committed_self_play(
    output: &Path,
    generation: usize,
) -> Result<(), Box<dyn Error>> {
    let manifest_path = output.join("pending-self-play.toml");
    if !manifest_path.exists() {
        return Ok(());
    }
    let pending = toml::from_str::<PendingSelfPlay>(&fs::read_to_string(&manifest_path)?)?;
    if pending.generation <= generation {
        fs::remove_file(manifest_path)?;
        let validation_path = validation_replay_path(output, pending.generation);
        if validation_path.exists() {
            fs::remove_file(validation_path)?;
        }
    }
    Ok(())
}

pub(super) fn recover_self_play(
    manifest_path: &Path,
    training_path: &Path,
    validation_path: &Path,
    generation: usize,
) -> Result<Option<GenerationSelfPlay>, Box<dyn Error>> {
    if !manifest_path.exists() {
        return Ok(None);
    }
    let pending = toml::from_str::<PendingSelfPlay>(&fs::read_to_string(manifest_path)?)?;
    if pending.generation != generation {
        return Err("pending self-play generation does not match run state".into());
    }
    if !training_path.exists() || !validation_path.exists() {
        return Err("pending self-play manifest references missing replay data".into());
    }
    let training = read_replay(training_path)?;
    let validation = read_replay(validation_path)?;
    if pending.training_samples != training.len() || pending.validation_samples != validation.len()
    {
        return Err("pending self-play manifest does not match replay data".into());
    }
    Ok(Some(GenerationSelfPlay {
        training,
        validation,
        pending,
        recovered: true,
    }))
}

pub(super) fn acquire_run_lock(output: &Path) -> io::Result<RunLock> {
    let path = suffixed_path(output, ".lock");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)?;
    FileExt::try_lock_exclusive(&file).map_err(|error| {
        if error.kind() == io::ErrorKind::WouldBlock {
            io::Error::new(
                error.kind(),
                format!("experiment output is already locked: {}", output.display()),
            )
        } else {
            error
        }
    })?;
    Ok(RunLock { _file: file })
}

pub(super) fn prepare_staging(output: &Path) -> io::Result<PathBuf> {
    let staging = suffixed_path(output, ".initializing");
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(staging.join("replay"))?;
    Ok(staging)
}

fn suffixed_path(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

pub(super) fn clean_output(output: &Path) -> Result<(), Box<dyn Error>> {
    if !output.exists() {
        return Ok(());
    }
    if verify_run_marker(output).is_err() {
        return Err(format!(
            "refusing to clean unrecognized output directory: {}",
            output.display()
        )
        .into());
    }
    fs::remove_dir_all(output)?;
    Ok(())
}

pub(super) fn append_metrics(
    path: &Path,
    metrics: &GenerationMetrics,
) -> Result<(), Box<dyn Error>> {
    let file = OpenOptions::new().append(true).open(path)?;
    let write_header = file.metadata()?.len() == 0;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(write_header)
        .from_writer(file);
    writer.serialize(metrics)?;
    writer.flush()?;
    Ok(())
}

pub(super) fn validate_metrics(path: &Path, generation: usize) -> Result<(), Box<dyn Error>> {
    if fs::metadata(path)?.len() == 0 {
        return if generation == 0 {
            Ok(())
        } else {
            Err("metrics are empty for a committed run".into())
        };
    }

    let mut reader = csv::Reader::from_path(path)?;
    if reader.headers()?.clone() != metrics_headers()? {
        return Err("metrics schema does not match the current Etive format".into());
    }
    let mut rows = 0;
    for (index, row) in reader.deserialize::<GenerationMetrics>().enumerate() {
        let row = row?;
        if row.generation != index + 1 {
            return Err("metrics generations are not contiguous".into());
        }
        rows += 1;
    }
    if rows != generation {
        return Err("metrics do not contain exactly one row per committed generation".into());
    }
    Ok(())
}

fn metrics_headers() -> Result<csv::StringRecord, Box<dyn Error>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.serialize(GenerationMetrics::default())?;
    let bytes = writer.into_inner()?;
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    Ok(reader.headers()?.clone())
}

pub(super) fn atomic_network_save(
    network: &OthelloNetwork,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    atomic_file_save(path, |temporary| {
        network.save(temporary)?;
        Ok(())
    })
}

pub(super) fn atomic_optimizer_save(
    trainer: &TrainingSession,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    atomic_file_save(path, |temporary| {
        trainer.save_optimizer(temporary)?;
        Ok(())
    })
}

pub(super) fn atomic_toml_save(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let source = toml::to_string(value)?;
    atomic_file_save(path, |temporary| {
        fs::write(temporary, source)?;
        Ok(())
    })
}

pub(super) fn verify_run_marker(output: &Path) -> io::Result<()> {
    if fs::read_to_string(output.join(".etive-run"))? != RUN_MARKER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Etive run marker",
        ));
    }
    Ok(())
}

pub(super) fn validate_run_state(state: RunState) -> io::Result<()> {
    if state.champion_generation > state.generation
        || !state.elapsed_seconds.is_finite()
        || state.elapsed_seconds < 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid run state",
        ));
    }
    Ok(())
}

pub(super) fn checkpoint_path(output: &Path, generation: usize) -> PathBuf {
    output.join(format!("generation-{generation:04}.burnpack"))
}

pub(super) fn optimizer_path(output: &Path, generation: usize) -> PathBuf {
    output.join(format!("generation-{generation:04}-optimizer.burnpack"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Outcome};
    use crate::othello::Board;
    use crate::othello::replay::{atomic_replay_save, replay_path};

    #[test]
    fn run_state_requires_current_explicit_fields() {
        let state: RunState =
            toml::from_str("generation = 4\nchampion_generation = 2\nelapsed_seconds = 300.0\n")
                .unwrap();
        assert!(validate_run_state(state).is_ok());

        assert!(toml::from_str::<RunState>("generation = 4\nelapsed_seconds = 300.0\n").is_err());
        assert!(
            toml::from_str::<RunState>(
                "generation = 4\nchampion_generation = 2\nnetwork_generation = 4\nelapsed_seconds = 300.0\n"
            )
            .is_err()
        );

        let future: RunState =
            toml::from_str("generation = 4\nchampion_generation = 5\nelapsed_seconds = 300.0\n")
                .unwrap();
        assert!(validate_run_state(future).is_err());
        assert!(
            validate_run_state(RunState {
                elapsed_seconds: f64::NAN,
                ..state
            })
            .is_err()
        );
        assert!(
            validate_run_state(RunState {
                elapsed_seconds: -1.0,
                ..state
            })
            .is_err()
        );
    }

    #[test]
    fn metrics_are_typed_and_match_committed_generations() {
        let path = std::env::temp_dir().join(format!("etive-metrics-{}.csv", std::process::id()));
        if path.exists() {
            fs::remove_file(&path).unwrap();
        }
        File::create(&path).unwrap();
        append_metrics(
            &path,
            &GenerationMetrics {
                generation: 1,
                checkpoint: "generation-0001.burnpack".into(),
                ..GenerationMetrics::default()
            },
        )
        .unwrap();

        validate_metrics(&path, 1).unwrap();
        assert!(validate_metrics(&path, 0).is_err());
        let header = fs::read_to_string(&path).unwrap();
        assert!(header.starts_with("generation,samples,training_samples"));
        assert!(header.contains("candidate_wins,baseline_wins"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn clean_only_removes_recognized_runs() {
        let output = std::env::temp_dir().join(format!("etive-clean-{}", std::process::id()));
        if output.exists() {
            fs::remove_dir_all(&output).unwrap();
        }
        fs::create_dir(&output).unwrap();

        assert!(clean_output(&output).is_err());
        assert!(output.exists());

        fs::write(output.join(".etive-run"), RUN_MARKER).unwrap();
        fs::write(
            output.join("state.toml"),
            "generation = 0\nchampion_generation = 0\nelapsed_seconds = 0.0\n",
        )
        .unwrap();
        clean_output(&output).unwrap();
        assert!(!output.exists());
    }

    #[test]
    fn run_lock_rejects_a_second_owner() {
        let output = std::env::temp_dir().join(format!("etive-lock-{}", std::process::id()));
        let lock_path = suffixed_path(&output, ".lock");
        if lock_path.exists() {
            fs::remove_file(&lock_path).unwrap();
        }

        let first = acquire_run_lock(&output).unwrap();
        let error = match acquire_run_lock(&output) {
            Ok(_) => panic!("second run unexpectedly acquired the output lock"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("already locked"));

        drop(first);
        drop(acquire_run_lock(&output).unwrap());
        fs::remove_file(lock_path).unwrap();
    }

    #[test]
    fn staging_replaces_interrupted_initialization() {
        let output = std::env::temp_dir().join(format!("etive-staging-{}", std::process::id()));
        let staging = suffixed_path(&output, ".initializing");
        if output.exists() {
            fs::remove_dir_all(&output).unwrap();
        }
        if staging.exists() {
            fs::remove_dir_all(&staging).unwrap();
        }
        fs::create_dir(&staging).unwrap();
        fs::write(staging.join("partial"), []).unwrap();

        let staging = prepare_staging(&output).unwrap();
        assert!(!staging.join("partial").exists());
        assert!(!output.exists());
        assert!(staging.join("replay").is_dir());

        fs::remove_dir_all(staging).unwrap();
    }

    #[test]
    fn recovery_requires_a_matching_committed_manifest() {
        let output = std::env::temp_dir().join(format!("etive-recovery-{}", std::process::id()));
        if output.exists() {
            fs::remove_dir_all(&output).unwrap();
        }
        let replay = output.join("replay");
        fs::create_dir_all(&replay).unwrap();
        let manifest_path = output.join("pending-self-play.toml");
        let training_path = replay_path(&output, 1);
        let validation_path = validation_replay_path(&output, 1);
        let mut policy = [0.0; Board::ACTION_COUNT];
        policy[19] = 1.0;
        let sample = SelfPlaySample {
            position: Board::default(),
            policy,
            outcome: Outcome::Draw,
            game: 1,
        };
        atomic_replay_save(std::slice::from_ref(&sample), &training_path).unwrap();

        assert!(
            recover_self_play(&manifest_path, &training_path, &validation_path, 1)
                .unwrap()
                .is_none()
        );

        atomic_replay_save(&[], &validation_path).unwrap();
        atomic_toml_save(
            &manifest_path,
            &PendingSelfPlay {
                generation: 1,
                training_samples: 1,
                validation_samples: 0,
                elapsed_seconds: 1.0,
                evaluations: 2,
                unique_games: 1,
            },
        )
        .unwrap();
        let recovered =
            recover_self_play(&manifest_path, &training_path, &validation_path, 1).unwrap();
        assert_eq!(recovered.unwrap().training.len(), 1);

        discard_committed_self_play(&output, 1).unwrap();
        assert!(!manifest_path.exists());
        assert!(!validation_path.exists());
        assert!(training_path.exists());

        fs::remove_dir_all(output).unwrap();
    }
}
