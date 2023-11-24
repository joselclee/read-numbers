#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>

#define IMG_SIZE 784
#define TRAINING_SIZE 5000
#define TESTING_SIZE 1500
#define NUM_CLASSES 10

class CDigit {
    private:
        unsigned char data[IMG_SIZE];
        char label;

    public: 
        unsigned char* Data() {
            return data;
        }
        char& Label() {
            return label;
        }
        double EuclideanDistance(const CDigit& i2) {
            double distance = 0;
            for (int i = 0 ; i < IMG_SIZE; i++) {
                double temp = data[i] - i2.data[i];
                distance += temp * temp;
            }
            return distance;
        }
        unsigned char& operator[](int i) {
            return data[i];
        }
};

class Classifier {
    private:
        int k = 3;

        std::vector<CDigit> training_data;
        std::vector<CDigit> testing_data;
        std::vector<char> classification;

        //Called by LoadTrainingData and LoadTestData

        void ReadDigits(const char* filename, std::vector<CDigit>& data) {
            std::ifstream file(filename, std::ios::binary);
            if (!file) {
                std::cerr << "File: " << filename << " not found.\n";
                return;
            }
            for (int i = 0 ; i < data.size(); i++) {
                char label;
                file.read(&label, sizeof(char));
                data[i].Label() = label;

                unsigned char* imgData = data[i].Data();
                file.read(reinterpret_cast<char*>(imgData), IMG_SIZE * sizeof(unsigned char));
            }
            file.close();
        }
    public:

        Classifier(int K = 3) : k{K} {
            training_data.resize(TRAINING_SIZE);
            testing_data.resize(TESTING_SIZE);
        }

        Classifier& operator = (const Classifier&) = delete;
        Classifier (const Classifier&) = delete;

        ~Classifier() = default;

        void LoadTrainingData(const char* filename) {
            ReadDigits(filename, training_data);
            std::random_shuffle(training_data.begin(), training_data.end());
        }
        void LoadTestData (const char* filename) {
            ReadDigits(filename, testing_data);
        }
        CDigit* TrainingData() {
            return training_data.data();
        }
        CDigit* TestingData() {
            return testing_data.data();
        }
        void Classify() {
            for (int i = 0; i < TESTING_SIZE; i++) {
                std::vector<std::pair<double, char>> dist_label;
                for (int j = 0; j < TRAINING_SIZE; j++) {
                    double dist = testing_data[i].EuclideanDistance(training_data[j]);
                    dist_label.push_back(std::make_pair(dist, training_data[j].Label()));
                }
                std::sort(dist_label.begin(), dist_label.end());
                std::vector<int> digit_frequency(NUM_CLASSES, 0);
                for (int n = 0; n < k; n++) {
                    digit_frequency[dist_label[n].second - '0']++;
                }
                int max = 0; 
                for (int n = 1; n < NUM_CLASSES; n++) {
                    if (digit_frequency[n] > digit_frequency[max]) {
                        max = n;
                    }
                }
                classification.push_back(static_cast<char>(max + '0'));
            }
        }
        std::vector<char>& Classification() {
            return classification;
        }
};

int main(int argc, char* argv[]) {
    Classifier c(10);
    const char* train_filename = "mnist_train_5000.csv";
    const char* test_filename = "mnist_test_1500.csv";

    c.LoadTestData(test_filename);
    c.LoadTrainingData(train_filename);

    c.Classify();

    int matches = 0;
    for (int i = 0; i < TESTING_SIZE; i++) {
        if (c.Classification()[i] == c.TestingData()[i].Label())
            matches++;
    }
    std::cout << "matches / testing_size = " << (double)matches / TESTING_SIZE << std::endl;

    char stop;
    std::cin >> stop;
    return 0;
}