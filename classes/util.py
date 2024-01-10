from datetime import datetime
import os

def debug( message ):
	now = datetime.now()

	print( now.strftime( "[%Y-%m-%d %H:%M:%S] " ) + message )

def get_data_folder( subfolder ):
	return os.path.dirname( __file__ ) + "/data/" + subfolder + "/"
