from pytorch_dl.core.logging import setup_logging, get_logger


setup_logging()

logger = get_logger(__name__)


logger.info(f"hello")