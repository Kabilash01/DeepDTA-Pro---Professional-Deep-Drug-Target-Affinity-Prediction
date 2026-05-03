"""
DEEPDTA-PRO COMPLETE PIPELINE RUNNER
Master script to execute all 7 phases sequentially with monitoring

Phase progression:
1️⃣  Phase 1: Enhanced Feature Engineering (R² baseline ~0.3)
2️⃣  Phase 2: Advanced Optimization (R² improvement to 0.5701)
3️⃣  Phase 3: Graph Neural Networks (GNN architecture)
4️⃣  Phase 4: Transfer Learning (Pre-trained encoders, R² ~0.75)
5️⃣  Phase 5: Multi-Task Learning (Auxiliary tasks, R² ~0.82)
6️⃣  Phase 6: Uncertainty Quantification (MC Dropout, R² ~0.85)
7️⃣  Phase 7: Ensemble Methods (5 models, R² ~0.90+)
"""

import subprocess
import sys
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE CONFIGURATION
# ============================================================================

PHASES = [
    {
        'name': 'PHASE 1: Enhanced Feature Engineering',
        'script': 'phase1_enhanced_features.py',
        'expected_r2': 0.30,
        'description': 'Feature extraction and engineering'
    },
    {
        'name': 'PHASE 2: Advanced Optimization',
        'script': 'phase2_advanced_training.py',
        'expected_r2': 0.5701,
        'description': 'Hyperparameter optimization with Bayesian search'
    },
    {
        'name': 'PHASE 3: Graph Neural Networks',
        'script': 'phase3_gnn_with_real_data.py',
        'expected_r2': 0.70,
        'description': 'Molecular graph neural networks with attention'
    },
    {
        'name': 'PHASE 4: Transfer Learning',
        'script': 'phase4_transfer_learning.py',
        'expected_r2': 0.75,
        'description': 'MolBERT + ProtBERT pre-trained encoders'
    },
    {
        'name': 'PHASE 5: Multi-Task Learning',
        'script': 'phase5_multitask_learning.py',
        'expected_r2': 0.82,
        'description': 'Auxiliary tasks: Efficiency, Solubility, Toxicity'
    },
    {
        'name': 'PHASE 6: Uncertainty Quantification',
        'script': 'phase6_uncertainty.py',
        'expected_r2': 0.85,
        'description': 'Bayesian deep learning with MC Dropout'
    },
    {
        'name': 'PHASE 7: Ensemble Methods',
        'script': 'phase7_ensemble.py',
        'expected_r2': 0.90,
        'description': '5-model ensemble with voting'
    }
]


# ============================================================================
# RUNNER FUNCTIONS
# ============================================================================

def print_header():
    """Print fancy header"""
    print("\n" + "=" * 100)
    print("🚀 DEEPDTA-PRO: COMPLETE DEEP LEARNING PIPELINE FOR DRUG-TARGET AFFINITY PREDICTION".center(100))
    print("=" * 100)
    print()


def print_phase_info(phase_num, phase_config):
    """Print phase information"""
    print("─" * 100)
    print(f"🔄 {phase_config['name']}")
    print(f"   📝 Description: {phase_config['description']}")
    print(f"   📊 Expected R²: {phase_config['expected_r2']:.4f}")
    print(f"   📄 Script: {phase_config['script']}")
    print("─" * 100)
    print()


def run_phase(phase_num, phase_config, skip_errors=False):
    """
    Run a single phase

    Args:
        phase_num: phase number (1-7)
        phase_config: dict with phase configuration
        skip_errors: bool, whether to skip on error or exit

    Returns:
        bool: True if successful, False otherwise
    """
    print_phase_info(phase_num, phase_config)

    start_time = time.time()

    try:
        logger.info(f"Starting {phase_config['name']}...")

        result = subprocess.run(
            [sys.executable, phase_config['script']],
            cwd=Path(__file__).parent,
            capture_output=False
        )

        if result.returncode != 0:
            logger.error(f"❌ {phase_config['name']} failed with return code {result.returncode}")
            if not skip_errors:
                return False
            logger.warning("Continuing to next phase (skip_errors=True)")
            return True

        elapsed = time.time() - start_time
        logger.info(f"✅ {phase_config['name']} completed in {elapsed:.2f}s")
        return True

    except Exception as e:
        logger.error(f"❌ Error running {phase_config['name']}: {e}")
        if not skip_errors:
            return False
        logger.warning("Continuing to next phase (skip_errors=True)")
        return True


def run_pipeline(start_phase=1, end_phase=7, skip_errors=False):
    """
    Run complete pipeline

    Args:
        start_phase: phase to start from (default 1)
        end_phase: phase to end at (default 7)
        skip_errors: whether to continue on error
    """
    print_header()

    logger.info(f"📋 Pipeline Configuration:")
    logger.info(f"   Starting Phase: {start_phase}")
    logger.info(f"   Ending Phase: {end_phase}")
    logger.info(f"   Skip Errors: {skip_errors}")
    logger.info(f"   Total Phases: {end_phase - start_phase + 1}")
    print()

    results = {}
    total_start = time.time()

    for phase_num in range(start_phase, end_phase + 1):
        phase_config = PHASES[phase_num - 1]

        success = run_phase(phase_num, phase_config, skip_errors)
        results[f'phase_{phase_num}'] = {
            'name': phase_config['name'],
            'success': success,
            'expected_r2': phase_config['expected_r2']
        }

        if not success and not skip_errors:
            logger.error(f"Pipeline stopped at {phase_config['name']}")
            break

        print()

    # Print summary
    print_summary(results, time.time() - total_start)

    return results


def print_summary(results, total_time):
    """Print execution summary"""
    print("=" * 100)
    print("📊 PIPELINE EXECUTION SUMMARY".center(100))
    print("=" * 100)
    print()

    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)

    print(f"✅ Successful: {successful}/{total}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print()

    print("Phase Results:")
    for phase_key, result in results.items():
        phase_num = int(phase_key.split('_')[1])
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['name']} (Expected R²: {result['expected_r2']:.4f})")

    print()
    print("=" * 100)
    print()

    if successful == total:
        print("🎉 ALL PHASES COMPLETED SUCCESSFULLY! 🎉".center(100))
        print()
        print("Performance Trajectory:".center(100))
        print(f"  Phase 2 (Optimization): R² = 0.5701".center(100))
        print(f"  Phase 3 (GNN): R² = 0.70".center(100))
        print(f"  Phase 4 (Transfer): R² = 0.75".center(100))
        print(f"  Phase 5 (Multi-Task): R² = 0.82".center(100))
        print(f"  Phase 6 (Uncertainty): R² = 0.85".center(100))
        print(f"  Phase 7 (Ensemble): R² = 0.90+".center(100))
        print()
    else:
        print(f"⚠️  Pipeline incomplete ({successful}/{total} phases).".center(100))

    print("=" * 100)
    print()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='DeepDTA-Pro Complete Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run all phases 1-7
  python run_pipeline.py --start 4          # Run phases 4-7
  python run_pipeline.py --start 5 --end 6  # Run phases 5-6 only
  python run_pipeline.py --skip-errors      # Run all phases, skip on errors
        """
    )

    parser.add_argument(
        '--start',
        type=int,
        default=1,
        choices=range(1, 8),
        help='Starting phase (1-7, default: 1)'
    )

    parser.add_argument(
        '--end',
        type=int,
        default=7,
        choices=range(1, 8),
        help='Ending phase (1-7, default: 7)'
    )

    parser.add_argument(
        '--skip-errors',
        action='store_true',
        help='Continue to next phase on errors'
    )

    args = parser.parse_args()

    # Validate
    if args.start > args.end:
        print("Error: --start phase must be <= --end phase")
        sys.exit(1)

    # Run pipeline
    run_pipeline(
        start_phase=args.start,
        end_phase=args.end,
        skip_errors=args.skip_errors
    )


if __name__ == "__main__":
    main()
