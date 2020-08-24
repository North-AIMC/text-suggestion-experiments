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

class NullPipeline(object):
    def __init__(self):
        pass

    def get_suggestions(self, text, group):
        return(['','',''])
