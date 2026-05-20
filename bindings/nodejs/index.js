const { Pipeline, PipelineError, Session, Result } = require("./lib/pipeline");

module.exports = {
    Pipeline,
    PipelineError,
    Session,
    Result,
    version: () => Pipeline.version(),
    listModules: () => Pipeline.listModules(),
    listPresets: () => Pipeline.listPresets(),
    setLogLevel: (level) => Pipeline.setLogLevel(level),
};
