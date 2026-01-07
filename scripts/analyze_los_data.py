#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOS Guidance Data Analyzer

Program untuk menganalisis data log dari LOS Guidance trajectory following.

Fitur:
1. Load dan parse CSV data
2. Hitung metrik error (RMSE, MAE, Max Error, etc.)
3. Plot 2D: XY trajectory, errors over time, velocity profiles
4. Plot 3D: 3D trajectory visualization
5. Comparative analysis: desired vs LOS reference vs actual

Usage:
    python analyze_los_data.py <csv_file>
    python analyze_los_data.py  # Auto-select latest file

Author: Generated for MPC path following analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import glob
from datetime import datetime


class LOSDataAnalyzer:
    def __init__(self, csv_file=None):
        """
        Initialize analyzer with CSV file.
        
        Args:
            csv_file: Path to CSV file. If None, auto-select latest file.
        """
        self.csv_file = csv_file
        self.data = None
        self.metrics = {}
        
        # Default log directory
        self.log_dir = os.path.expanduser("~/catkin_ws_ros/src/mpc_offboard/logs")
        
        # Output directory for plots
        self.output_dir = os.path.expanduser("~/catkin_ws_ros/src/mpc_offboard/analysis")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load CSV data."""
        if self.csv_file is None:
            # Auto-select latest file
            files = glob.glob(os.path.join(self.log_dir, "los_log_*.csv"))
            if not files:
                raise FileNotFoundError(f"No log files found in {self.log_dir}")
            self.csv_file = max(files, key=os.path.getctime)
            print(f"📂 Auto-selected latest file: {os.path.basename(self.csv_file)}")
        
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"File not found: {self.csv_file}")
        
        self.data = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(self.data)} data points from {os.path.basename(self.csv_file)}")
        
        # Filter out data before tracking starts (if needed)
        # Keep only data during TRACK mode (not WAIT_ALT or FINISHED holding)
        if 'status' in self.data.columns:
            # Filter TRACK mode or when cross_track is not zero
            pass  # Keep all data for now
    
    def compute_metrics(self):
        """Compute error metrics."""
        print("\n" + "="*60)
        print("📊 COMPUTING ERROR METRICS")
        print("="*60)
        
        # Error from desired path (path asli)
        error_desired_xy = self.data['error_from_desired_xy'].values
        error_desired_3d = self.data['error_from_desired_3d'].values
        error_desired_x = self.data['error_from_desired_x'].values
        error_desired_y = self.data['error_from_desired_y'].values
        error_desired_z = self.data['error_from_desired_z'].values
        
        # Error from LOS reference
        error_los_xy = self.data['error_xy'].values
        error_los_3d = self.data['error_3d'].values
        
        # Cross-track error dari LOS
        cross_track = self.data['cross_track_error'].values
        
        # Compute metrics
        self.metrics = {
            # Error from Desired Path
            'desired_rmse_xy': np.sqrt(np.mean(error_desired_xy**2)),
            'desired_rmse_3d': np.sqrt(np.mean(error_desired_3d**2)),
            'desired_mae_xy': np.mean(np.abs(error_desired_xy)),
            'desired_mae_3d': np.mean(np.abs(error_desired_3d)),
            'desired_max_xy': np.max(np.abs(error_desired_xy)),
            'desired_max_3d': np.max(np.abs(error_desired_3d)),
            'desired_std_xy': np.std(error_desired_xy),
            'desired_std_3d': np.std(error_desired_3d),
            
            # Per-axis errors from desired
            'desired_rmse_x': np.sqrt(np.mean(error_desired_x**2)),
            'desired_rmse_y': np.sqrt(np.mean(error_desired_y**2)),
            'desired_rmse_z': np.sqrt(np.mean(error_desired_z**2)),
            'desired_max_x': np.max(np.abs(error_desired_x)),
            'desired_max_y': np.max(np.abs(error_desired_y)),
            'desired_max_z': np.max(np.abs(error_desired_z)),
            
            # Error from LOS Reference
            'los_rmse_xy': np.sqrt(np.mean(error_los_xy**2)),
            'los_rmse_3d': np.sqrt(np.mean(error_los_3d**2)),
            'los_mae_xy': np.mean(np.abs(error_los_xy)),
            'los_max_xy': np.max(np.abs(error_los_xy)),
            
            # Cross-track error
            'cross_track_rmse': np.sqrt(np.mean(cross_track**2)),
            'cross_track_mae': np.mean(np.abs(cross_track)),
            'cross_track_max': np.max(np.abs(cross_track)),
            'cross_track_std': np.std(cross_track),
        }
        
        # Print metrics
        print("\n📍 Error from DESIRED PATH (path asli):")
        print(f"   RMSE XY    : {self.metrics['desired_rmse_xy']:.4f} m")
        print(f"   RMSE 3D    : {self.metrics['desired_rmse_3d']:.4f} m")
        print(f"   MAE XY     : {self.metrics['desired_mae_xy']:.4f} m")
        print(f"   MAE 3D     : {self.metrics['desired_mae_3d']:.4f} m")
        print(f"   Max XY     : {self.metrics['desired_max_xy']:.4f} m")
        print(f"   Max 3D     : {self.metrics['desired_max_3d']:.4f} m")
        print(f"   Std XY     : {self.metrics['desired_std_xy']:.4f} m")
        
        print(f"\n   Per-axis RMSE: X={self.metrics['desired_rmse_x']:.4f}, "
              f"Y={self.metrics['desired_rmse_y']:.4f}, Z={self.metrics['desired_rmse_z']:.4f} m")
        print(f"   Per-axis Max:  X={self.metrics['desired_max_x']:.4f}, "
              f"Y={self.metrics['desired_max_y']:.4f}, Z={self.metrics['desired_max_z']:.4f} m")
        
        print("\n🎯 Error from LOS REFERENCE (lookahead point):")
        print(f"   RMSE XY    : {self.metrics['los_rmse_xy']:.4f} m")
        print(f"   RMSE 3D    : {self.metrics['los_rmse_3d']:.4f} m")
        print(f"   MAE XY     : {self.metrics['los_mae_xy']:.4f} m")
        print(f"   Max XY     : {self.metrics['los_max_xy']:.4f} m")
        
        print("\n📐 Cross-track Error:")
        print(f"   RMSE       : {self.metrics['cross_track_rmse']:.4f} m")
        print(f"   MAE        : {self.metrics['cross_track_mae']:.4f} m")
        print(f"   Max        : {self.metrics['cross_track_max']:.4f} m")
        print(f"   Std        : {self.metrics['cross_track_std']:.4f} m")
        
        return self.metrics
    
    def plot_2d_trajectory(self, save=True, show=True):
        """Plot 2D XY trajectory comparison."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Convert to numpy arrays
        desired_x = self.data['desired_x'].values
        desired_y = self.data['desired_y'].values
        los_ref_x = self.data['los_ref_x'].values
        los_ref_y = self.data['los_ref_y'].values
        actual_x = self.data['actual_x'].values
        actual_y = self.data['actual_y'].values
        
        # Plot desired path
        ax.plot(desired_x, desired_y, 
                'g-', linewidth=2, label='Desired Path', alpha=0.8)
        
        # Plot LOS reference
        ax.plot(los_ref_x, los_ref_y, 
                'b--', linewidth=1.5, label='LOS Reference', alpha=0.7)
        
        # Plot actual trajectory
        ax.plot(actual_x, actual_y, 
                'r-', linewidth=1.5, label='Actual', alpha=0.9)
        
        # Mark start and end
        ax.scatter(actual_x[0], actual_y[0],
                   c='green', s=100, marker='o', zorder=5, label='Start')
        ax.scatter(actual_x[-1], actual_y[-1],
                   c='red', s=100, marker='x', zorder=5, label='End')
        
        ax.set_xlabel('X (m) - East', fontsize=12)
        ax.set_ylabel('Y (m) - North', fontsize=12)
        ax.set_title('2D Trajectory Comparison (XY Plane)', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add metrics annotation
        metrics_text = (f"RMSE (from desired): {self.metrics['desired_rmse_xy']:.3f} m\n"
                       f"Max Error: {self.metrics['desired_max_xy']:.3f} m\n"
                       f"Cross-track RMSE: {self.metrics['cross_track_rmse']:.3f} m")
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'trajectory_2d_xy.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_3d_trajectory(self, save=True, show=True):
        """Plot 3D trajectory comparison."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert to numpy arrays
        desired_x = self.data['desired_x'].values
        desired_y = self.data['desired_y'].values
        desired_z = self.data['desired_z'].values
        los_ref_x = self.data['los_ref_x'].values
        los_ref_y = self.data['los_ref_y'].values
        los_ref_z = self.data['los_ref_z'].values
        actual_x = self.data['actual_x'].values
        actual_y = self.data['actual_y'].values
        actual_z = self.data['actual_z'].values
        
        # Plot desired path
        ax.plot(desired_x, desired_y, desired_z,
                'g-', linewidth=2, label='Desired Path', alpha=0.8)
        
        # Plot LOS reference
        ax.plot(los_ref_x, los_ref_y, los_ref_z,
                'b--', linewidth=1.5, label='LOS Reference', alpha=0.7)
        
        # Plot actual trajectory
        ax.plot(actual_x, actual_y, actual_z,
                'r-', linewidth=1.5, label='Actual', alpha=0.9)
        
        # Mark start and end
        ax.scatter(actual_x[0], actual_y[0], actual_z[0],
                   c='green', s=100, marker='o', label='Start')
        ax.scatter(actual_x[-1], actual_y[-1], actual_z[-1],
                   c='red', s=100, marker='x', label='End')
        
        ax.set_xlabel('X (m) - East', fontsize=11)
        ax.set_ylabel('Y (m) - North', fontsize=11)
        ax.set_zlabel('Z (m) - Up', fontsize=11)
        ax.set_title('3D Trajectory Comparison', fontsize=14)
        ax.legend(loc='best', fontsize=9)
        
        # Equal aspect ratio for 3D
        max_range = np.array([
            actual_x.max() - actual_x.min(),
            actual_y.max() - actual_y.min(),
            actual_z.max() - actual_z.min()
        ]).max() / 2.0
        
        mid_x = (actual_x.max() + actual_x.min()) * 0.5
        mid_y = (actual_y.max() + actual_y.min()) * 0.5
        mid_z = (actual_z.max() + actual_z.min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'trajectory_3d.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_errors_over_time(self, save=True, show=True):
        """Plot errors over time."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        
        # Convert to numpy arrays
        time = self.data['elapsed_time'].values
        error_x = self.data['error_from_desired_x'].values
        error_y = self.data['error_from_desired_y'].values
        error_z = self.data['error_from_desired_z'].values
        error_xy = self.data['error_from_desired_xy'].values
        error_3d = self.data['error_from_desired_3d'].values
        cross_track = self.data['cross_track_error'].values
        along_track = self.data['along_track_distance'].values
        
        # Plot 1: Position errors from desired path
        ax1 = axes[0]
        ax1.plot(time, error_x, 'r-', label='Error X', alpha=0.8)
        ax1.plot(time, error_y, 'g-', label='Error Y', alpha=0.8)
        ax1.plot(time, error_z, 'b-', label='Error Z', alpha=0.8)
        ax1.set_ylabel('Position Error (m)', fontsize=10)
        ax1.set_title('Position Errors from Desired Path', fontsize=12)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Plot 2: XY and 3D error magnitude
        ax2 = axes[1]
        ax2.plot(time, error_xy, 'b-', label='Error XY', linewidth=1.5)
        ax2.plot(time, error_3d, 'r-', label='Error 3D', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=self.metrics['desired_rmse_xy'], color='b', linestyle='--', 
                    label=f'RMSE XY: {self.metrics["desired_rmse_xy"]:.3f}m', alpha=0.7)
        ax2.set_ylabel('Error Magnitude (m)', fontsize=10)
        ax2.set_title('Error Magnitude from Desired Path', fontsize=12)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cross-track error
        ax3 = axes[2]
        ax3.plot(time, cross_track, 'purple', linewidth=1.5)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax3.axhline(y=self.metrics['cross_track_rmse'], color='r', linestyle='--', 
                    label=f'RMSE: {self.metrics["cross_track_rmse"]:.3f}m', alpha=0.7)
        ax3.axhline(y=-self.metrics['cross_track_rmse'], color='r', linestyle='--', alpha=0.7)
        ax3.fill_between(time, -self.metrics['cross_track_rmse'], self.metrics['cross_track_rmse'], 
                         alpha=0.2, color='red')
        ax3.set_ylabel('Cross-track Error (m)', fontsize=10)
        ax3.set_title('Cross-track Error (+ = right, - = left)', fontsize=12)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Along-track distance
        ax4 = axes[3]
        ax4.plot(time, along_track, 'green', linewidth=1.5)
        ax4.set_xlabel('Time (s)', fontsize=11)
        ax4.set_ylabel('Along-track Distance (m)', fontsize=10)
        ax4.set_title('Along-track Progress', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'errors_over_time.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_velocity_profiles(self, save=True, show=True):
        """Plot velocity profiles."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Convert to numpy arrays
        time = self.data['elapsed_time'].values
        ref_vx = self.data['ref_vx'].values
        ref_vy = self.data['ref_vy'].values
        ref_vz = self.data['ref_vz'].values
        actual_vx = self.data['actual_vx'].values
        actual_vy = self.data['actual_vy'].values
        actual_vz = self.data['actual_vz'].values
        
        # Plot 1: Velocity X
        ax1 = axes[0]
        ax1.plot(time, ref_vx, 'b--', label='Reference Vx', alpha=0.8)
        ax1.plot(time, actual_vx, 'r-', label='Actual Vx', alpha=0.8)
        ax1.set_ylabel('Vx (m/s)', fontsize=10)
        ax1.set_title('Velocity X (East)', fontsize=12)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Velocity Y
        ax2 = axes[1]
        ax2.plot(time, ref_vy, 'b--', label='Reference Vy', alpha=0.8)
        ax2.plot(time, actual_vy, 'r-', label='Actual Vy', alpha=0.8)
        ax2.set_ylabel('Vy (m/s)', fontsize=10)
        ax2.set_title('Velocity Y (North)', fontsize=12)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Velocity Z
        ax3 = axes[2]
        ax3.plot(time, ref_vz, 'b--', label='Reference Vz', alpha=0.8)
        ax3.plot(time, actual_vz, 'r-', label='Actual Vz', alpha=0.8)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Vz (m/s)', fontsize=10)
        ax3.set_title('Velocity Z (Up)', fontsize=12)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'velocity_profiles.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_altitude_profile(self, save=True, show=True):
        """Plot altitude profile."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Convert to numpy arrays
        time = self.data['elapsed_time'].values
        desired_z = self.data['desired_z'].values
        los_ref_z = self.data['los_ref_z'].values
        actual_z = self.data['actual_z'].values
        error_z = self.data['error_from_desired_z'].values
        
        # Plot 1: Altitude comparison
        ax1 = axes[0]
        ax1.plot(time, desired_z, 'g-', label='Desired', linewidth=2, alpha=0.8)
        ax1.plot(time, los_ref_z, 'b--', label='LOS Reference', linewidth=1.5, alpha=0.7)
        ax1.plot(time, actual_z, 'r-', label='Actual', linewidth=1.5, alpha=0.9)
        ax1.set_ylabel('Altitude (m)', fontsize=10)
        ax1.set_title('Altitude Profile', fontsize=12)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Altitude error
        ax2 = axes[1]
        ax2.plot(time, error_z, 'purple', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.axhline(y=self.metrics['desired_rmse_z'], color='r', linestyle='--', 
                    label=f'RMSE: {self.metrics["desired_rmse_z"]:.3f}m', alpha=0.7)
        ax2.axhline(y=-self.metrics['desired_rmse_z'], color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Altitude Error (m)', fontsize=10)
        ax2.set_title('Altitude Error from Desired Path', fontsize=12)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'altitude_profile.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_yaw_profile(self, save=True, show=True):
        """Plot yaw angle profile."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Convert to numpy arrays
        time = self.data['elapsed_time'].values
        
        # Convert to degrees
        desired_yaw_deg = np.degrees(self.data['desired_yaw'].values)
        los_ref_yaw_deg = np.degrees(self.data['los_ref_yaw'].values)
        actual_yaw_deg = np.degrees(self.data['actual_yaw'].values)
        
        # Plot 1: Yaw comparison
        ax1 = axes[0]
        ax1.plot(time, desired_yaw_deg, 'g-', label='Desired', linewidth=2, alpha=0.8)
        ax1.plot(time, los_ref_yaw_deg, 'b--', label='LOS Reference', linewidth=1.5, alpha=0.7)
        ax1.plot(time, actual_yaw_deg, 'r-', label='Actual', linewidth=1.5, alpha=0.9)
        ax1.set_ylabel('Yaw (deg)', fontsize=10)
        ax1.set_title('Yaw Angle Profile', fontsize=12)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Yaw error
        yaw_error = los_ref_yaw_deg - actual_yaw_deg
        # Wrap to [-180, 180]
        yaw_error = np.arctan2(np.sin(np.radians(yaw_error)), np.cos(np.radians(yaw_error)))
        yaw_error = np.degrees(yaw_error)
        
        ax2 = axes[1]
        ax2.plot(time, yaw_error, 'purple', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Yaw Error (deg)', fontsize=10)
        ax2.set_title('Yaw Error (LOS Reference - Actual)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'yaw_profile.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_error_distribution(self, save=True, show=True):
        """Plot error distribution histograms."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Convert to numpy arrays
        error_xy = self.data['error_from_desired_xy'].values
        error_3d = self.data['error_from_desired_3d'].values
        cross_track = self.data['cross_track_error'].values
        error_z = self.data['error_from_desired_z'].values
        
        # Plot 1: XY error from desired path
        ax1 = axes[0, 0]
        ax1.hist(error_xy, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax1.axvline(x=self.metrics['desired_rmse_xy'], color='r', linestyle='--', 
                    label=f'RMSE: {self.metrics["desired_rmse_xy"]:.3f}m')
        ax1.axvline(x=self.metrics['desired_mae_xy'], color='g', linestyle='--', 
                    label=f'MAE: {self.metrics["desired_mae_xy"]:.3f}m')
        ax1.set_xlabel('XY Error (m)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('XY Error Distribution (from Desired Path)', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 3D error from desired path
        ax2 = axes[0, 1]
        ax2.hist(error_3d, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(x=self.metrics['desired_rmse_3d'], color='r', linestyle='--', 
                    label=f'RMSE: {self.metrics["desired_rmse_3d"]:.3f}m')
        ax2.set_xlabel('3D Error (m)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('3D Error Distribution (from Desired Path)', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cross-track error
        ax3 = axes[1, 0]
        ax3.hist(cross_track, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax3.axvline(x=self.metrics['cross_track_rmse'], color='r', linestyle='--', 
                    label=f'RMSE: {self.metrics["cross_track_rmse"]:.3f}m')
        ax3.axvline(x=-self.metrics['cross_track_rmse'], color='r', linestyle='--')
        ax3.set_xlabel('Cross-track Error (m)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Cross-track Error Distribution', fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Altitude error
        ax4 = axes[1, 1]
        ax4.hist(error_z, bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax4.axvline(x=self.metrics['desired_rmse_z'], color='r', linestyle='--', 
                    label=f'RMSE: {self.metrics["desired_rmse_z"]:.3f}m')
        ax4.axvline(x=-self.metrics['desired_rmse_z'], color='r', linestyle='--')
        ax4.set_xlabel('Altitude Error (m)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('Altitude Error Distribution', fontsize=11)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'error_distribution.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_all(self, save=True, show=True):
        """Generate all plots."""
        print("\n" + "="*60)
        print("📈 GENERATING PLOTS")
        print("="*60)
        
        self.plot_2d_trajectory(save=save, show=show)
        self.plot_3d_trajectory(save=save, show=show)
        self.plot_errors_over_time(save=save, show=show)
        self.plot_velocity_profiles(save=save, show=show)
        self.plot_altitude_profile(save=save, show=show)
        self.plot_yaw_profile(save=save, show=show)
        self.plot_error_distribution(save=save, show=show)
        
        print(f"\n✅ All plots saved to: {self.output_dir}")
    
    def save_metrics_to_csv(self):
        """Save metrics to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f'metrics_{timestamp}.csv')
        
        # Convert metrics dict to DataFrame
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(filename, index=False)
        
        print(f"💾 Metrics saved to: {filename}")
    
    def generate_report(self):
        """Generate full analysis report."""
        print("\n" + "="*60)
        print("📋 LOS GUIDANCE TRAJECTORY ANALYSIS REPORT")
        print("="*60)
        print(f"Data file: {os.path.basename(self.csv_file)}")
        print(f"Data points: {len(self.data)}")
        print(f"Duration: {self.data['elapsed_time'].iloc[-1]:.2f} seconds")
        
        self.compute_metrics()
        self.plot_all(save=True, show=False)
        self.save_metrics_to_csv()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print(f"📁 Output directory: {self.output_dir}")
        print("="*60)


def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = None  # Auto-select latest
    
    try:
        analyzer = LOSDataAnalyzer(csv_file)
        analyzer.generate_report()
        
        # Show plots interactively at the end
        print("\n🖼️  Displaying plots... Close windows to exit.")
        analyzer.plot_all(save=False, show=True)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
