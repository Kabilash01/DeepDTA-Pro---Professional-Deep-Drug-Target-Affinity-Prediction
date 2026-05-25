"""
DEEPDTA-PRO COMPLETE GML PIPELINE RUNNER
Master script to execute all 7 Graph Machine Learning phases sequentially.

Phase progression:
1  Phase 1: Graph Feature Engineering    - molecular/protein graph analysis
2  Phase 2: GCN Baseline                 - Graph Convolutional Network
3  Phase 3: GAT                          - Graph Attention Network
4  Phase 4: JK-GIN                       - Jumping Knowledge GIN
5  Phase 5: Multi-Task GNN               - shared encoder, multiple heads
6  Phase 6: Bayesian GNN                 - MC Dropout uncertainty
7  Phase 7: GNN Ensemble                 - GCN + GAT + GIN heterogeneous
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
        'name': 'PHASE 1: Graph Feature Engineering',
        'script': 'phase1_enhanced_features.py',
        'expected_r2': None,
        'description': 'Molecular graph analysis: atom/bond features, degree distribution'
    },
    {
        'name': 'PHASE 2: GCN Baseline',
        'script': 'phase2_advanced_training.py',
        'expected_r2': 0.55,
        'description': 'Graph Convolutional Network with residual connections'
    },
    {
        'name': 'PHASE 3: GAT',
        'script': 'phase3_gnn_with_real_data.py',
        'expected_r2': 0.60,
        'description': 'Graph Attention Network - multi-head attention over atoms'
    },
    {
        'name': 'PHASE 4: JK-GIN',
        'script': 'phase4_transfer_learning.py',
        'expected_r2': 0.65,
        'description': 'Jumping Knowledge GIN - most expressive GNN in WL hierarchy'
    },
    {
        'name': 'PHASE 5: Multi-Task GNN',
        'script': 'phase5_multitask_learning.py',
        'expected_r2': 0.68,
        'description': 'Shared GIN encoder with affinity + efficiency + selectivity heads'
    },
    {
        'name': 'PHASE 6: Bayesian GNN',
        'script': 'phase6_uncertainty.py',
        'expected_r2': 0.70,
        'description': 'MC Dropout uncertainty quantification on GIN'
    },
    {
        'name': 'PHASE 7: GNN Ensemble',
        'script': 'phase7_ensemble.py',
        'expected_r2': 0.75,
        'description': 'Heterogeneous ensemble: GCN + GAT + GIN (5 members)'
    }
]


# ============================================================================
# RUNNER FUNCTIONS
# ============================================================================

def print_header():
    """Print fancy header"""
    print("\n" + "=" * 100)
    print("DEEPDTA-PRO: COMPLETE GML PIPELINE FOR DRUG-TARGET AFFINITY PREDICTION".center(100))
    print("=" * 100)
    print()


def print_phase_info(phase_num, phase_config):
    """Print phase information"""
    print("-" * 100)
    print(f"[PHASE {phase_num}] {phase_config['name']}")
    print(f"   Description: {phase_config['description']}")
    r2 = phase_config['expected_r2']
    r2_str = f"{r2:.4f}" if r2 is not None else "N/A (analysis phase)"
    print(f"   Expected R2: {r2_str}")
    print(f"   Script: {phase_config['script']}")
    print("-" * 100)
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
            logger.error(f"FAILED: {phase_config['name']} (return code {result.returncode})")
            if not skip_errors:
                return False
            logger.warning("Continuing to next phase (skip_errors=True)")
            return True

        elapsed = time.time() - start_time
        logger.info(f"DONE: {phase_config['name']} in {elapsed:.2f}s")
        return True

    except Exception as e:
        logger.error(f"ERROR running {phase_config['name']}: {e}")
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

    logger.info(f"Pipeline Configuration:")
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
    print("PIPELINE EXECUTION SUMMARY".center(100))
    print("=" * 100)
    print()

    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)

    print(f"Successful: {successful}/{total}")
    print(f"Total Time: {total_time:.2f}s")
    print()

    print("Phase Results:")
    for phase_key, result in results.items():
        phase_num = int(phase_key.split('_')[1])
        status = "OK  " if result['success'] else "FAIL"
        r2 = result['expected_r2']
        r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
        print(f"  [{status}] {result['name']} (Expected R2: {r2_str})")

    print()
    print("=" * 100)
    print()

    if successful == total:
        print("ALL PHASES COMPLETED SUCCESSFULLY!".center(100))
        print()
        print("GML Performance Trajectory (targets):".center(100))
        print("  Phase 2 (GCN):         R2 ~ 0.55".center(100))
        print("  Phase 3 (GAT):         R2 ~ 0.60".center(100))
        print("  Phase 4 (JK-GIN):      R2 ~ 0.65".center(100))
        print("  Phase 5 (Multi-Task):  R2 ~ 0.68".center(100))
        print("  Phase 6 (Bayesian):    R2 ~ 0.70".center(100))
        print("  Phase 7 (Ensemble):    R2 ~ 0.75".center(100))
        print()
    else:
        print(f"Pipeline incomplete ({successful}/{total} phases).".center(100))

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
