import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== Concrete CDP Material Definition (Compression Only) ===")
print("Note: All values are based on Eurocode 2 and ABAQUS CDP documentation.\n")

# Step 1: Ask user for f_ck
fck = float(input("Enter the characteristic compressive strength f_ck (in MPa): "))

# Step 2: Calculate f_cm (mean compressive strength)
f_cm = fck + 8
print(f"\nMean compressive strength f_cm = {f_cm:.2f} MPa (calculated as f_ck + 8)")

# Step 3: Calculate E_cm (modulus of elasticity)
E_cm_GPa = 22 * ((f_cm / 10) ** 0.3)
E_cm_MPa = E_cm_GPa * 1000
print(f"Modulus of elasticity E_cm = {E_cm_GPa:.3f} GPa")
print(f"Modulus of elasticity E_cm = {E_cm_MPa:.0f} MPa")

# Step 4: Calculate strain at peak stress (ε_c1)
eps_c1 = 0.7 * (f_cm ** 0.31) * 0.001
if eps_c1 > 0.0028:
    eps_c1 = 0.0028

# Round to 4 decimal digits and store it
eps_c1 = round(eps_c1, 4)

# Print with 4 decimal digits
print(f"Strain at peak stress ε_c1 = {eps_c1:.4f}")


# Step 5: Calculate ultimate compressive strain (ε_cu1)
if fck < 50:
    eps_cu1 = 0.0035
else:
    eps_cu1 = (2.8 + 27 * ((98 - f_cm) / 100) ** 4) * 0.001
print(f"Ultimate compressive strain ε_cu1 = {eps_cu1:.6f}")

# Step 6: Generate strain values from 0 to ε_cu1 with step 0.0001
strain_values = np.arange(0, eps_cu1 + 0.0001, 0.0001)
print(f"\nGenerated {len(strain_values)} strain points from 0 to ε_cu1.")

# Step 7: Calculate η = strain / ε_c1
eta_values = strain_values / eps_c1

# Step 8: Calculate k using E_cm in MPa
k = 1.05 * E_cm_MPa * abs(eps_c1) / f_cm
print(f"Nonlinear curve shape parameter k = {k:.4f}")

# Step 9: Directly calculate stress using the Eurocode equation
stress_values = f_cm * ((k * eta_values - eta_values ** 2) / (1 + (k - 2) * eta_values))

# Step 10: Calculate elastic strain
elastic_strain_values = stress_values / E_cm_MPa

# Step 11: Calculate inelastic strain
inelastic_strain_values = np.zeros_like(strain_values)
mask = strain_values > eps_c1
inelastic_strain_values[mask] = strain_values[mask] - elastic_strain_values[mask]

# Step 12: Calculate damage parameter
damage_values = np.zeros_like(strain_values)
damage_values[mask] = 1 - (stress_values[mask] / f_cm)

# Step 13: Calculate plastic strain
plastic_strain_values = np.zeros_like(strain_values)
plastic_strain_values[mask] = (
    inelastic_strain_values[mask]
    - (damage_values[mask] / (1 - damage_values[mask])) * (stress_values[mask] / E_cm_MPa)
)

# Step 14: Ensure non-negative and monotonically increasing values
def ensure_non_negative_and_monotonic(arr):
    arr = np.maximum(arr, 0)
    return np.maximum.accumulate(arr)

inelastic_strain_values = ensure_non_negative_and_monotonic(inelastic_strain_values)
plastic_strain_values = ensure_non_negative_and_monotonic(plastic_strain_values)
damage_values = ensure_non_negative_and_monotonic(damage_values)

# Step 15: Save stress vs inelastic strain starting from ε_c1
start_index = np.argmax(strain_values >= eps_c1)
export_data = pd.DataFrame({
    "Stress [MPa]": stress_values[start_index:],
    "Inelastic Strain [-]": inelastic_strain_values[start_index:]
})
export_data.to_csv("plastic_compression_behaviour.csv", index=False)
print("Saved 'plastic_compression_behaviour.csv' with stress vs inelastic strain from ε_c1 onward.")

# Step 16: Save damage vs inelastic strain starting from ε_c1
export_damage_data = pd.DataFrame({
    "Damage Parameter [-]": damage_values[start_index:],
    "Inelastic Strain [-]": inelastic_strain_values[start_index:]
})
export_damage_data.to_csv("damage_inelastic_compression.csv", index=False)
print("Saved 'damage_inelastic_compression.csv' with damage vs inelastic strain from ε_c1 onward.")

# Step 17: Plot stress–strain relationship
plt.figure(figsize=(8, 5))
plt.plot(strain_values, stress_values, label="Stress-Strain Curve", color="steelblue")
plt.xlabel("Strain [-]")
plt.ylabel("Stress [MPa]")
plt.title("Concrete Stress–Strain Relationship (Eurocode)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Step 18: Plot damage parameter vs strain
plt.figure(figsize=(8, 5))
plt.plot(strain_values, damage_values, label="Damage Parameter", color="crimson")
plt.xlabel("Strain [-]")
plt.ylabel("Damage Parameter [-]")
plt.title("Damage vs. Strain")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Step 19 (updated): Plot total, elastic, inelastic, and plastic strain in one figure
plt.figure(figsize=(9, 5.5))
plt.plot(strain_values, strain_values, label="Total Strain", color="black", linestyle="--")
plt.plot(strain_values, elastic_strain_values, label="Elastic Strain", color="green")
plt.plot(strain_values, inelastic_strain_values, label="Inelastic Strain", color="orange")
plt.plot(strain_values, plastic_strain_values, label="Plastic Strain", color="blue")
plt.xlabel("Strain [-]")
plt.ylabel("Strain [-]")
plt.title("Strain Components vs. Total Strain")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
