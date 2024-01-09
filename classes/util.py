def debug( message ):
	now = datetime.now()

	print( now.strftime( '[%Y-%m-%d %H:%M:%S] ' ) + message )
