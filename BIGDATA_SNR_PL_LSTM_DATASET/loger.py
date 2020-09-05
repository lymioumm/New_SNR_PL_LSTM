#
#
# import logging
# import logging.config
#
#
# class SingleLevelFilter(object):
#     def __init__(self, pass_level):
#         self.pass_level = pass_level
#
#     def filter(self, record):
#         if self.pass_level == record.levelno:
#             return True
#         return False
#
#
# LEVEL_COLOR = {
#     logging.DEBUG: '\33[2;39m',
#     logging.INFO: '\33[0;37m',
#     logging.WARN: '\33[4;35m',
#     logging.ERROR: '\33[5;31m',
#     logging.FATAL: '\33[7;31m'
# }
#
#
# # class ScreenHandler(logging.StreamHandler):
# #     def emit(self, record):
# #         try:
# #             msg = self.format(record)
# #             stream = self.stream
# #             fs = LEVEL_COLOR[record.levelno] + "%s\n" + '\33[0m'
# #             try:
# #                 if isinstance(msg, unicode) and getattr(stream, 'encoding', None):
# #                     ufs = fs.decode(stream.encoding)
# #                     try:
# #                         stream.write(ufs % msg)
# #                     except UnicodeEncodeError:
# #                         stream.write((ufs % msg).encode(stream.encoding))
# #                 else:
# #                     stream.write(fs % msg)
# #             except UnicodeError:
# #                 stream.write(fs % msg.encode("UTF-8"))
# #
# #             self.flush()
# #         except (KeyboardInterrupt, SystemExit):
# #             raise
# #         except:
# #             self.handleError(record)
#
#
# def init_logger():
#     conf = {'version': 1,
#             'disable_existing_loggers': True,
#             'incremental': False,
#             'formatters': {'myformat1': {'class': 'logging.Formatter',
#                                          'format': '|%(asctime)s|%(name)s|%(filename)s|%(lineno)d|%(levelname)s|%(message)s',
#                                          'datefmt': '%Y-%m-%d %H:%M:%S'}
#                            },
#             # 'filters': {'filter_by_name': {'class': 'logging.Filter',
#             #                                'name': 'logger_for_filter_name'},
#             #
#             #             'filter_single_level_pass': {'()': 'mylogger.SingleLevelFilter',
#             #                                          'pass_level': logging.WARN}
#             #             },
#             'handlers': {'console': {'class': 'logging.StreamHandler',
#                                      'level': 'INFO',
#                                      'formatter': 'myformat1',
#                                      'filters': ['filter_single_level_pass', ]},
#
#                          'screen': {'()': 'mylogger.ScreenHandler',
#                                     'level': logging.INFO,
#                                     'formatter': 'myformat1',
#                                     'filters': ['filter_by_name', ]}
#                          },
#             'loggers': {'logger_for_filter_name': {'handlers': ['console', 'screen'],
#                                                    'filters': ['filter_by_name', ],
#                                                    'level': 'INFO'},
#                         'logger_for_all': {'handlers': ['console', ],
#                                            'filters': ['filter_single_level_pass', ],
#                                            'level': 'INFO',
#                                            'propagate': False}
#                         }
#             }
#     logging.config.dictConfig(conf)
#
#
# if __name__ == '__main__':
#     init_logger()
#     logger_for_filter_name = logging.getLogger('logger_for_filter_name')
#     logger_for_filter_name.debug('logger_for_filter_name')
#     logger_for_filter_name.info('logger_for_filter_name')
#     logger_for_filter_name.warn('logger_for_filter_name')
#     logger_for_filter_name.error('logger_for_filter_name')
#     logger_for_filter_name.critical('logger_for_filter_name')
#
#     logger_for_all = logging.getLogger('logger_for_all')
#     logger_for_all.debug('logger_for_all')
#     logger_for_all.info('logger_for_all')
#     logger_for_all.warn('logger_for_all')
#     logger_for_all.error('logger_for_all')
#     logger_for_all.critical('logger_for_all')

import logging
from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=20000,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
if __name__ == '__main__':
    log = Logger('all.log',level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    Logger('error.log', level='error').logger.error('error')