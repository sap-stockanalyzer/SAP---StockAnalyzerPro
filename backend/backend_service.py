# Admin consolidated router import
from .admin_consolidated_router import admin_consolidated_router  

# Commented out routers
# from .page_data_router import page_data_router  
# from .settings_consolidated_router import settings_consolidated_router

ROUTERS = [
    admin_consolidated_router,
    # page_data_router,
    # settings_consolidated_router,
]