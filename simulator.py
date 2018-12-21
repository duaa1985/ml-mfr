import numpy as np
import csv

##========================================================================
##--------------- Constant Parameters------------------------------------
##========================================================================

# planck's constant in j.s
H = 6.626e-34 

# speed of light in m/s
C = 299792458

# lambda in m
LAMBDA = 1550e-9

# frequency in HZ
V = C/LAMBDA

# chosen OSNR noise BW in m
BN_db = 0.1 * 10**-9
BN = V**2 * BN_db # HZ

##==== OSNR AND MARGIN VALUES FOR EACH MODULATION FORMAT======
THRESHOLD_OSNR = [12.6, 13, 16, 21] # OSNR_BPSK, OSNR_QPSK, OSNR_16QAM, OSNR_64QAM
MARGIN = [0.5, 1, 2, 3]             # MARGIN_BPSK, MARGIN_QPSK, MARGIN_16QAM, MARGIN_64QAM
CLASSIFICATION = [2, 4, 16, 64]     # class 0, class 1, class 2, class 3 | 4 is no_class

## ========================================================================
##------------------Calculating OSNR---------------------------------------
## ========================================================================

class Simulator:

    def __init__(self):
        self.set_features()

    def set_features(self):

        self.symbol_rate     = np.random.randint(30, 91)
        self.roll_off        = np.random.randint(1, 6)/100
        self.channel_load    = np.random.randint(1, 121)
        self.dispersion      = np.random.uniform(4, 21)
        self.nonlinear_index = np.random.uniform(0.8, 1.6)
        self.loss            = np.random.uniform(0.15, 0.2)
        self.span_count      = np.random.randint(1, 51)
        self.span_length     = np.random.randint(40000, 120001)/1000
        self.noise_figure    = np.random.uniform(4, 6.5)

        self.status_linear   = False

    def get_features(self):

        features = [self.symbol_rate, self.roll_off, self.launch_power, self.channel_load, self.dispersion, self.nonlinear_index, self.loss, self.span_count, self.span_length, self.noise_figure, self.channel_grid]
        return features

    def convert_to_linear(self):

        if self.status_linear is True:
            return
        else:
            self.symbol_rate *= 1e9
            self.dispersion *= 1e-6
            self.nonlinear_index *= 1e-3
            self.loss = ((self.loss*1e-5)/4.34)*1e2
            self.span_length *= 1e3
            self.noise_figure = 10**(self.noise_figure/10)

            self.status_linear = True

    def calculate_osnr(self):

        #Defined Terms:
        gain = self.loss * self.span_length

        loss_eff = 1/(2*self.loss)
        self.channel_grid = (1 + self.roll_off) * self.symbol_rate
        bandwidth_wdm = self.channel_grid * self.channel_load

        #Noise Power
        noise_power =  H * V * BN * (gain * self.noise_figure - 1)  

        #Launch Power:
        x = 2 * 16 * (self.nonlinear_index**2) * self.span_length *  BN * np.arcsinh(((1/3)*(np.pi**2))*self.dispersion*self.span_length*bandwidth_wdm**2)

        epsilon = 0.3 * np.log(1 + ((6*loss_eff)/(self.span_length*np.arcsinh(((np.pi**2)/2) * self.dispersion * loss_eff * bandwidth_wdm**2))))
        y = self.span_count**epsilon

        self.launch_power = self.symbol_rate * ((27 * np.pi * self.dispersion * noise_power) / (x*y))**(1/3)

        #OSNR:
        osnr_total = (2/3)*(self.launch_power/(noise_power*self.span_count))

        #Convert to dB:
        osnr_total_dB = 10 * np.log10(osnr_total)

        return osnr_total_dB

    def classify(self, osnr_total_dB):

        classification = 4
        for i in reversed(range(4)):
             L = osnr_total_dB - (THRESHOLD_OSNR[i])
             if 0 < L <  MARGIN[i]:
                return i
        return classification

    def simulate(self, nb_samples=1, output_path=None):

        all_samples = []
        while (len(all_samples) < nb_samples):

            #Set features
            self.set_features()
            self.convert_to_linear()

            #Simulate
            osnr_total_dB = self.calculate_osnr()

            #Classify
            classification = self.classify(osnr_total_dB)
            if classification == 4:
                continue

            #Data:
            features = self.get_features()
            data = [*features, classification]
            all_samples.append(data)
            
        with open(output_path, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['Symbol Rate', 'Roll-off', 'Launch Power', 'Channel load', 'Dispersion', 'Nonlinear index', 'Loss', 'Span count', 'Span lenght', 'Noise Figure', 'Channel grid', 'Modulation Classification'])
            spamwriter.writerows(all_samples)


if __name__ == "__main__":
    
    simulator = Simulator()
    simulator.simulate(100000, output_path="data.csv")