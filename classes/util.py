from datetime import datetime
import os

def debug( message, eol = True, timestamp = True ):
	now = datetime.now()

	print( ( now.strftime( "[%Y-%m-%d %H:%M:%S] " ) if timestamp else "" ) + message, end = "\n" if eol else "", flush = True )

def get_file_path( file_name ):
	return os.path.dirname( __file__ ) + "/../data/" + file_name
