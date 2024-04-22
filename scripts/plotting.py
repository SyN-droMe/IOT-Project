import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read BPM values from the text file
with open("cr1.txt", "r") as file:
    bpm_values = [float(line.strip()) for line in file]

# Step 2: Calculate statistics
highest_bpm = max(bpm_values)
lowest_bpm = min(bpm_values)

# Step 3: Calculate moving average
window_size = 860
moving_avg = np.convolve(bpm_values, np.ones(window_size)/window_size, mode='valid')

# Step 4: Plot the data
plt.figure(figsize=(10, 6))
plt.plot(bpm_values, label='BPM')
plt.plot(range(window_size-1, len(bpm_values)), moving_avg, color='orange', linestyle='--', label='Moving Average')
plt.title('BPM Values Over Time with Moving Average')
plt.xlabel('Time')
plt.ylabel('BPM')

# Step 5: Mark the highest and lowest values
plt.scatter(bpm_values.index(highest_bpm), highest_bpm, color='red', label='Highest BPM')
plt.scatter(bpm_values.index(lowest_bpm), lowest_bpm, color='green', label='Lowest BPM')

plt.legend()
plt.grid(True)
plt.show()

with open("cr1.txt", "r") as file:
    bpm_values = [float(line.strip()) for line in file]

highest_bpm = max(bpm_values)
lowest_bpm = min(bpm_values)
average_bpm = np.mean(moving_avg)

print("Highest BPM:", highest_bpm)
print("Lowest BPM:", lowest_bpm)
print("Moving Average BPM:", average_bpm)