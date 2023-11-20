import torch
from torch.utils.data import Sampler

MAX_EXTRA_WEIGHT = 100  # Example value, adjust as needed
TARGET_WIN_RATE = 0.5   # This is an example value for scaling the weight
MAX_INDICES = 10000     # Example value, adjust as needed

class PerformanceBasedSampler(Sampler):
    def __init__(self, data_source, performance_metrics):
        # print("Initializing PerformanceBasedSampler")
        self.data_source = data_source
        self.performance_metrics = performance_metrics
        self.indices = self.generate_indices()
        # print(f"Initial indices generated: {len(self.indices)}")

    def generate_indices(self):
        epsilon = 1e-6  # A small value to avoid division by zero
        indices = []
        extra_weights_distribution = {}  # For debugging

        for idx, data in enumerate(self.data_source):
            word_length = len(data[3])
            performance = self.performance_metrics.get(word_length, {})
            win_rate = performance.get('win_rate', 1)

            # Calculate extra weight based on win rate
            extra_weight = max(1, min(int(TARGET_WIN_RATE \
                / (win_rate + epsilon)), MAX_EXTRA_WEIGHT))
            indices.extend([idx] * extra_weight)

            # Collect data for debugging
            extra_weights_distribution.setdefault(extra_weight, 0)
            extra_weights_distribution[extra_weight] += 1

            # Debug line for every 100 entries
            # if idx % 100 == 0:
                # print(f"Processed {idx} entries, current index count: {len(indices)}")

        # Limit the total number of indices
        if len(indices) > MAX_INDICES:
            indices = indices[:MAX_INDICES]

        # print(f"Total indices generated: {len(indices)}")
        # print(f"Extra weights distribution: {extra_weights_distribution}")
        return indices

    def __iter__(self):
        # print("Creating iterator for indices")
        return iter(self.indices)

    def __len__(self):
        # print(f"Length of sampler: {len(self.indices)}")
        return len(self.indices)

def update_sampler(sampler, new_performance_metrics):
    # print("Updating sampler with new performance metrics")
    sampler.performance_metrics = new_performance_metrics
    sampler.indices = sampler.generate_indices()
    # print(f"Updated indices count: {len(sampler.indices)}")













# import torch
# from torch.utils.data import Sampler

# MAX_EXTRA_WEIGHT = 100
# TARGET_WIN_RATE = 0.5

# def update_sampler(sampler, new_performance_metrics):
#     print("Updating sampler with new performance metrics")  # Debug line
#     sampler.performance_metrics = new_performance_metrics
#     sampler.indices = sampler.generate_indices()
#     print(f"Updated indices count: {len(sampler.indices)}")  # Debug line


# class PerformanceBasedSampler(Sampler):
#     def __init__(self, data_source, performance_metrics):
#         print("Initializing PerformanceBasedSampler")  # Debug line
#         self.data_source = data_source
#         self.performance_metrics = performance_metrics
#         self.indices = self.generate_indices()
#         print(f"Initial indices generated: {len(self.indices)}")  # Debug line


#     def generate_indices(self):
#         epsilon = 1e-6
#         indices = []
#         extra_weights_distribution = {}  # New

#         for idx, data in enumerate(self.data_source):
#             word_length = len(data[3])
#             performance = self.performance_metrics.get(word_length, {})
#             win_rate = performance.get('win_rate', 1)

#             extra_weight = max(1, min(int(1 / (win_rate + epsilon)), MAX_EXTRA_WEIGHT))  # Modified
#             indices.extend([idx] * extra_weight)

#             extra_weights_distribution.setdefault(extra_weight, 0)  # New
#             extra_weights_distribution[extra_weight] += 1  # New

#             if idx % 100 == 0:
#                 print(f"Processed {idx} entries, current index count: {len(indices)}")

#         # Limit total indices
#         if len(indices) > MAX_INDICES:
#             indices = indices[:MAX_INDICES]

#         print(f"Total indices generated: {len(indices)}")
#         print(f"Extra weights distribution: {extra_weights_distribution}")  # New

#         return indices

#     # def generate_indices(self):
#     #     epsilon = 1e-6  # A small value to avoid division by zero
#     #     indices = []

#     #     for idx, data in enumerate(self.data_source):
#     #         word_length = len(data[3])  # Assuming the original word is the fourth element in the tuple
#     #         performance = self.performance_metrics.get(word_length, {})
#     #         win_rate = performance.get('win_rate', 1)

#     #         # Adjust calculation to avoid division by zero
#     #         extra_weight = max(1, int(1 / (win_rate + epsilon)))

#     #         indices.extend([idx] * extra_weight)

#     #         # Debug lines for each iteration
#     #         if idx % 100 == 0:  # Adjust this value based on your dataset size
#     #             print(f"Processed {idx} entries, current index count: {len(indices)}")  # Debug line

#         print(f"Total indices generated: {len(indices)}")  # Debug line
#         return indices

#     def __iter__(self):
#         print("Creating iterator for indices")  # Debug line
#         return iter(self.indices)

#     def __len__(self):
#         print(f"Length of sampler: {len(self.indices)}")  # Debug line
#         return len(self.indices)

# # Rest of the code remains the same
