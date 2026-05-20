plugins {
    id("com.android.library")
    `maven-publish`
}

android {
    namespace = "io.vietasr"
    compileSdk = 34

    defaultConfig {
        minSdk = 24
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }
}

publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "io.vietasr"
            artifactId = "vietasr"
            version = "0.1.0"

            afterEvaluate {
                from(components["release"])
            }
        }
    }
}
