"""
MIT License

Copyright (c) 2024 Laciel Alonso Llanes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Contact Information:
# Author: Laciel Alonso Llanes
# Contact: laciel.alonso@gmail.com
# GitHub: https://github.com/LacielAlonsoLlanes

import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def __init__(self, export):
        self.export = export

    def histogram(self, distribution, save_filename):
        try:
            maximum = np.max(distribution)
            minimum = np.min(distribution)

            # Create the histogram
            plt.hist(distribution, bins=np.arange(minimum, maximum + 1.5) - 0.5, color='blue', rwidth=0.8, density=True)

            # Customize the plot
            plt.title('Histogram | Normalized')
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            # Show the histogram
            # plt.show()

            self.export.plot(save_filename)

            plt.close()

        except:
            pass

    def histogram_log_log(self, distribution, save_filename):
        try:
            log_data = np.log(distribution)

            maximum = np.max(log_data)
            minimum = np.min(log_data)

            # Create the histogram
            plt.hist(log_data, bins=np.arange(minimum, maximum + 0.1, 0.1), color='blue', rwidth=0.8, density=True)

            # Customize the plot
            plt.title('Histogram | Normalized')
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            # Show the histogram
            # plt.show()

            self.export.plot(save_filename)

            plt.close()
        except:
            pass

    def cfd_loglog(self, distribution, save_filename):
        try:
            maximum = np.max(distribution)
            minimum = np.min(distribution)

            # Calculate the CDF
            hist, bin_edges = np.histogram(self.distribution_stopped_times, bins=np.arange(minimum, maximum + 1.5) - 0.5, density=True)
            cdf = np.cumsum(hist)
            print("Last value of CDF:", cdf[-1])

            # Create the log-log scale CDF plot
            plt.loglog(bin_edges[1:], cdf, marker='o', linestyle='-', color='blue', label='CDF')

            # Customize the plot
            plt.title('Cumulative Distribution Function (CDF) | Normalized')
            plt.xlabel('t_s')
            plt.ylabel('Cumulative probability')
            plt.legend(loc='lower right')

            # Show the plot
            # plt.show()

            self.export.plot(save_filename)

            plt.close()
        except:
            pass

    def survival_loglog(self, distribution, save_filename):
        try:
            maximum = np.max(distribution)
            minimum = np.min(distribution)

            # Calculate the CDF
            hist, bin_edges = np.histogram(distribution, bins=np.arange(minimum, maximum + 1.5) - 0.5, density=True)
            cdf = np.cumsum(hist)
            print("Last value of CDF:", cdf[-1])

            # Create the log-log scale CDF plot
            plt.loglog(bin_edges[1:], 1 - cdf, marker='o', linestyle='-', color='blue', label='CDF')

            # Customize the plot
            plt.title('Cumulative Distribution Function (CDF)')
            plt.xlabel('Value')
            plt.ylabel('Cumulative probability')
            plt.legend(loc='lower right')

            # Show the plot
            # plt.show()

            self.export.plot(save_filename)

            data_to_save = np.column_stack((bin_edges[1:], 1 - cdf))
            np.savetxt("survival_loglog_" + save_filename + ".csv", data_to_save, delimiter=",", header="Value,Cumulative Probability", comments="")

            plt.close()
        except:
            pass

    def survival_semilog(self, distribution, save_filename):
        try:
            maximum = np.max(distribution)
            minimum = np.min(distribution)

            # Calculate the CDF
            hist, bin_edges = np.histogram(distribution, bins=np.arange(minimum, maximum + 1.5) - 0.5, density=True)
            cdf = np.cumsum(hist)
            print("Last value of CDF:", cdf[-1])

            # Create the CDF plot with linear x-axis and logarithmic y-axis
            plt.semilogy(bin_edges[1:], 1 - cdf, marker='o', linestyle='-', color='blue', label='CDF')

            # Customize the plot
            plt.title('Cumulative Distribution Function (CDF)')
            plt.xlabel('Value')
            plt.ylabel('Cumulative probability (log scale)')
            plt.legend(loc='lower right')

            # Show the plot
            # plt.show()

            self.export.plot(save_filename)

            data_to_save = np.column_stack((bin_edges[1:], 1 - cdf))
            np.savetxt("survival_semilog_" + save_filename + ".csv", data_to_save, delimiter=",", header="Value,Cumulative Probability", comments="")

            plt.close()
        except:
            pass

    def survival_semilog_interactive(self, distribution, color, label):
        try:
            maximum = np.max(distribution)
            minimum = np.min(distribution)

            # Calculate the CDF
            hist, bin_edges = np.histogram(distribution, bins=np.arange(minimum, maximum + 1.5) - 0.5, density=True)
            cdf = np.cumsum(hist)
            print("Last value of CDF:", cdf[-1])

            # Create the CDF plot with linear x-axis and logarithmic y-axis
            rbin_edges = bin_edges[1:].copy()
            rcfd = 1 - cdf.copy()

            return rbin_edges, rcfd

        except:
            pass

    def survival_loglog_interactive(self, distribution, color, label):
        try:
            maximum = np.max(distribution)
            minimum = np.min(distribution)

            # Calculate the CDF
            hist, bin_edges = np.histogram(distribution, bins=np.arange(minimum, maximum + 1.5) - 0.5, density=True)
            cdf = np.cumsum(hist)
            print("Last value of CDF:", cdf[-1])

            return bin_edges[1:], 1 - cdf

        except:
            pass

    @staticmethod
    def plot_customizer():
        # Customize the plot
        plt.title('Cumulative Distribution Function (CDF)')
        plt.xlabel('Value')
        plt.ylabel('Cumulative probability (log scale)')
        plt.legend(loc='lower right')

