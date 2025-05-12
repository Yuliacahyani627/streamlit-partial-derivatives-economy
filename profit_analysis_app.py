import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Judul ---
st.title("Aplikasi Turunan Parsial - Studi Kasus Ekonomi (Fungsi Keuntungan)")

# --- Definisi Variabel dan Fungsi ---
x, y = sp.symbols('x y')
f = 50*x + 40*y - 0.5*x**2 - 0.3*y**2 - 0.2*x*y

# --- Input Titik Evaluasi ---
x0 = st.number_input("Masukkan nilai x (unit produk A)", value=10.0)
y0 = st.number_input("Masukkan nilai y (unit produk B)", value=10.0)

# --- Turunan Parsial ---
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)

# --- Evaluasi nilai turunan di titik (x0, y0) ---
df_dx_val = df_dx.evalf(subs={x: x0, y: y0})
df_dy_val = df_dy.evalf(subs={x: x0, y: y0})
f_val = f.evalf(subs={x: x0, y: y0})

st.write(f"**Turunan Parsial ∂f/∂x:** {df_dx} → nilai di titik: {df_dx_val}")
st.write(f"**Turunan Parsial ∂f/∂y:** {df_dy} → nilai di titik: {df_dy_val}")

# --- Plot 3D Permukaan dan Bidang Singgung ---
X_vals = np.linspace(x0 - 5, x0 + 5, 50)
Y_vals = np.linspace(y0 - 5, y0 + 5, 50)
X, Y = np.meshgrid(X_vals, Y_vals)

# Konversi f(x, y) ke fungsi numpy
f_np = sp.lambdify((x, y), f, 'numpy')
Z = f_np(X, Y)

# Bidang singgung
Z_tangent = f_val + df_dx_val * (X - x0) + df_dy_val * (Y - y0)

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, label='Permukaan Fungsi')
ax.plot_surface(X, Y, Z_tangent, color='red', alpha=0.5, label='Bidang Singgung')
ax.scatter(x0, y0, f_val, color='black', s=50)

ax.set_title("Permukaan Fungsi Keuntungan dan Bidang Singgung")
ax.set_xlabel("Produk A (x)")
ax.set_ylabel("Produk B (y)")
ax.set_zlabel("Keuntungan f(x, y)")

st.pyplot(fig)
