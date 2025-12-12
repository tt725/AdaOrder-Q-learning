import logging
from logging import handlers

# ========================================================================= #
#                     the level of the l0gger                               #
#               debug < info < warning < error < crit                       #
# ========================================================================= #

class Logger():
    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.level_relations = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'crit': logging.CRITICAL
        }
        self.logger = logging.getLogger(filename)
        #set the style of the log
        format_str = logging.Formatter(fmt)
        #set the level of the log
        self.logger.setLevel(self.level_relations.get(level))
        #print the log to screen
        sh = logging.StreamHandler()
        #set the style of the screen print of the log
        sh.setFormatter(format_str)
        #write the log to the file
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')
        #set the style of write log to the file
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)
