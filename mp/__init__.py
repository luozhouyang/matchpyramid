from easylib.log import TimedRotatingLoggerInitializer

initializer = TimedRotatingLoggerInitializer(
    name='matchpyramid',
    path='matchpyramid.log',
    format='%(asctime)s - %(levelname)s - %(filename)10s - %(lineno)4d - %(message)s')
initializer.initialize()
