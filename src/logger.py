import logging
import os
import time

class Logger:
    """Logger class helper to log training progress to a file and the console.
    """
    def __init__(self, log_file, experiment_title):
        self.log_file = log_file
        self.experiment_title = experiment_title
        self.logger = self._setup_logger()

    def _setup_logger(self):
        # Create the log directory if it doesn't exist
        print(self.log_file)
        log_dir = os.path.dirname(self.log_file)
        os.makedirs(log_dir, exist_ok=True)

        # Set up logging configuration
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] %(message)s',
                            handlers=[
                                logging.FileHandler(self.log_file),
                                logging.StreamHandler()
                            ])

        # Add experiment title to the log file
        logging.getLogger().info('Experiment: {}'.format(self.experiment_title))

        # Return the logger object
        return logging.getLogger()

    def log_epoch(self, epoch, train_loss, val_loss):
        # Log epoch information
        self.logger.info("Epoch: {} | Train Loss: {:.4f} | Val Loss: {:.4f}".format(epoch, train_loss, val_loss))

    def log_time(self, epoch, elapsed_time):
        # Log time information
        self.logger.info("Time elapsed for epoch {}: {:.2f} seconds".format(epoch, elapsed_time))

    def close(self):
        # Close the logger to release any resources
        logging.shutdown()

# Usage example
# if __name__ == '__main__':
#     # Set up logger
#     log_file = 'logs/my_project.log'
#     experiment_title = 'Experiment 1'
#     logger = Logger(log_file, experiment_title)

#     # Log training progress
#     for epoch in range(1, 6):
#         start_time = time.time()

#         # Training code...
#         train_loss = 0.123
#         val_loss = 0.045

#         # Log epoch information
#         logger.log_epoch(epoch, train_loss, val_loss)

#         # Log time information
#         elapsed_time = time.time() - start_time
#         logger.log_time(epoch, elapsed_time)

#     # Close the logger to release any resources
#     logger.close()
