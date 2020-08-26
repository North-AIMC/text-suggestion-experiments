import time

class TimeTestPipeline(object):
    def __init__(self):
        pass

    def get_suggestions(self, text, group):
        if(group=='+'):
            time.sleep(0.002)
            return(['','',''])
        elif(group=='-'):
            time.sleep(0.001)
            return(['','',''])
        else:
            return(['','',''])

class StopWordTestPipeline(object):
    def __init__(self):
        pass

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(['the','you','love'])
        elif(group=='-'):
            return(['the','you','hate'])
        else:
            return(['','',''])

class BiasTestPipeline(object):
    def __init__(self):
        pass

    def get_suggestions(self, text, group):
        if(group=='+'):
            return(['Love','Happy','Joy'])
        elif(group=='-'):
            return(['Hate','Stupid','Pain'])
        else:
            return(['','',''])

class NullTestPipeline(object):
    def __init__(self):
        pass

    def get_suggestions(self, text, group):
        return(['','',''])
