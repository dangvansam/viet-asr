plugins {
    id("com.android.application")
}

android {
    namespace = "io.vietasr.demo"
    compileSdk = 34

    defaultConfig {
        applicationId = "io.vietasr.demo"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
}

dependencies {
    implementation(project(":lib"))
}
