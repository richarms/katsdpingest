#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()

katsdp.setDependencies([
    'ska-sa/katsdpsigproc/master',
    'ska-sa/katsdpdockerbase/master',
    'ska-sa/katsdpservices/master',
    'ska-sa/katsdptelstate/master',
    'ska-sa/katdal'])

catchError {
    katsdp.stagePrepare(python2: false, python3: true,
                        timeout: [time: 60, unit: 'MINUTES'])
    katsdp.stageNosetestsGpu(cuda: true, opencl: true)
    katsdp.stageFlake8()
    katsdp.stageMypy()
    katsdp.stageMakeDocker(venv: true)

    stage('katsdpingest/autotuning') {
        if (katsdp.notYetFailed()) {
            katsdp.simpleNode(label: 'cuda-GeForce_GTX_TITAN_X') {
                deleteDir()
                katsdp.unpackGit()
                katsdp.unpackVenv()
                katsdp.unpackKatsdpdockerbase()
                katsdp.virtualenv('venv3') {
                    dir('git') {
                        lock("katsdpingest-autotune-${env.BRANCH_NAME}") {
                            sh './jenkins-autotune.sh geforce_gtx_titan_x'
                        }
                    }
                }
            }
        }
    }
}
katsdp.mail('sdpdev+katsdpingest@ska.ac.za')
