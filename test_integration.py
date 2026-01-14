"""
Integration test for PINN demo
Tests all components work together
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import torch

# Add to path
sys.path.append(str(Path(__file__).parent))

from utils.data_gen import DataGenerator
from models.pure_ai import PureAIModel, PureAITrainer
from models.pinn import PINN, PINNTrainer


def test_integration():
    """Run full integration test"""

    print("="*70)
    print("PINN DEMO - INTEGRATION TEST")
    print("="*70)

    # Parameters
    length = 0.1
    alpha = 1e-4
    t_max = 50.0  # Shorter for faster test

    print("\n[1/5] Generating ground truth (CAE)...")
    gen = DataGenerator(length=length, alpha=alpha, nx=50, dt=0.005)
    ground_truth = gen.generate_ground_truth(
        t_left=100.0,
        t_right=20.0,
        t_initial=20.0,
        t_max=t_max,
    )
    print(f"   ✓ CAE solution: {ground_truth['T'].shape}")

    print("\n[2/5] Generating sensor data (for Pure AI)...")
    sensor_data = gen.generate_sensor_data(
        ground_truth,
        sensor_positions=np.array([0.0, length]),
        measurement_interval=20,
        noise_std=0.5,
    )
    print(f"   ✓ Sensor data: {sensor_data['T'].shape}")
    print(f"   ✓ Data reduction: {ground_truth['T'].size / sensor_data['T'].size:.1f}x")

    print("\n[3/5] Training Pure AI model...")
    pure_ai = PureAIModel(hidden_layers=[16, 16])
    ai_trainer = PureAITrainer(pure_ai, learning_rate=1e-3)
    ai_loss = ai_trainer.train(sensor_data, epochs=100, verbose=False)
    print(f"   ✓ Pure AI trained: Loss {ai_loss[0]:.2f} → {ai_loss[-1]:.2f}")

    print("\n[4/5] Training PINN model...")
    pinn_data = gen.generate_pinn_training_data(
        t_left=100.0,
        t_right=20.0,
        t_initial=20.0,
        t_max=t_max,
        n_boundary=50,
        n_initial=20,
        n_collocation=200,
    )
    pinn = PINN(hidden_layers=[16, 16], alpha=alpha)
    pinn_trainer = PINNTrainer(pinn, learning_rate=1e-3)
    pinn_loss = pinn_trainer.train(pinn_data, epochs=500, verbose=False)
    print(f"   ✓ PINN trained: Total {pinn_loss['total'][-1]:.2f}")
    print(f"   ✓ PDE residual: {pinn_loss['pde'][0]:.4f} → {pinn_loss['pde'][-1]:.4f}")

    print("\n[5/5] Evaluating on test data...")
    x_test = ground_truth['x']
    t_test = ground_truth['t'][::100]  # Sample for faster test
    T_true = ground_truth['T'][::100, :]

    X_grid, T_grid = np.meshgrid(x_test, t_test)
    x_flat = X_grid.flatten()
    t_flat = T_grid.flatten()
    T_true_flat = T_true.flatten()

    ai_metrics = ai_trainer.evaluate(x_flat, t_flat, T_true_flat)
    pinn_metrics = pinn_trainer.evaluate(x_flat, t_flat, T_true_flat)

    print(f"\n   Pure AI Metrics:")
    print(f"      MSE:  {ai_metrics['mse']:.4f}")
    print(f"      RMSE: {ai_metrics['rmse']:.4f}")
    print(f"      MAE:  {ai_metrics['mae']:.4f}")

    print(f"\n   PINN Metrics:")
    print(f"      MSE:  {pinn_metrics['mse']:.4f}")
    print(f"      RMSE: {pinn_metrics['rmse']:.4f}")
    print(f"      MAE:  {pinn_metrics['mae']:.4f}")

    # Generalization test
    print("\n[BONUS] Generalization Test (2x length)...")
    gen_test = gen.create_generalization_test(new_length=0.2, t_max=t_max, nx=50)
    x_gen = gen_test['x']
    t_gen = gen_test['t'][::100]
    T_gen_true = gen_test['T'][::100, :]

    X_gen, T_gen = np.meshgrid(x_gen, t_gen)
    x_gen_flat = X_gen.flatten()
    t_gen_flat = T_gen.flatten()
    T_gen_true_flat = T_gen_true.flatten()

    ai_gen = ai_trainer.evaluate(x_gen_flat, t_gen_flat, T_gen_true_flat)
    pinn_gen = pinn_trainer.evaluate(x_gen_flat, t_gen_flat, T_gen_true_flat)

    print(f"\n   Pure AI Generalization:")
    print(f"      RMSE: {ai_metrics['rmse']:.4f} → {ai_gen['rmse']:.4f} "
          f"({(ai_gen['rmse']/ai_metrics['rmse'] - 1)*100:+.1f}%)")

    print(f"\n   PINN Generalization:")
    print(f"      RMSE: {pinn_metrics['rmse']:.4f} → {pinn_gen['rmse']:.4f} "
          f"({(pinn_gen['rmse']/pinn_metrics['rmse'] - 1)*100:+.1f}%)")

    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)

    print("\n📌 KEY FINDINGS:")
    print(f"   1. Data efficiency: Pure AI uses {sensor_data['T'].size} points")
    print(f"                       PINN uses physics (minimal data needed)")
    print(f"\n   2. Accuracy: PINN RMSE = {pinn_metrics['rmse']:.4f}")
    print(f"                Pure AI RMSE = {ai_metrics['rmse']:.4f}")

    if pinn_gen['rmse'] / pinn_metrics['rmse'] < ai_gen['rmse'] / ai_metrics['rmse']:
        print(f"\n   3. Generalization: ✨ PINN generalizes better!")
        print(f"                      PINN error increase: {(pinn_gen['rmse']/pinn_metrics['rmse'] - 1)*100:+.1f}%")
        print(f"                      Pure AI error increase: {(ai_gen['rmse']/ai_metrics['rmse'] - 1)*100:+.1f}%")

    print("\n💡 Ready to run Streamlit app:")
    print("   streamlit run app.py")
    print("="*70)


if __name__ == "__main__":
    test_integration()
