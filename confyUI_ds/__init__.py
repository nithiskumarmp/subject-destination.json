from .document_scanner import DocumentScannerNode, SimpleDocumentScannerNode
from .black_bg_scanner import BlackBackgroundScannerNode

NODE_CLASS_MAPPINGS = {
    "DocumentScanner": DocumentScannerNode,
    "SimpleDocumentScanner": SimpleDocumentScannerNode,
    "BlackBackgroundScanner": BlackBackgroundScannerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DocumentScanner": "Document Scanner",
    "SimpleDocumentScanner": "Simple Document Scanner", 
    "BlackBackgroundScanner": "Black Background Scanner"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']