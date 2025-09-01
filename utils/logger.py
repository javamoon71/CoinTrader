import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("signals.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("SignalLogger")