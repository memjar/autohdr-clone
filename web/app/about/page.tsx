'use client'

export default function About() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
      </div>

      <div className="relative z-10 p-8 max-w-4xl mx-auto">
      {/* Navigation */}
      <nav className="flex justify-between items-center mb-12">
        <a href="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/25 group-hover:shadow-cyan-500/40 transition-all">
            <span className="text-xl">⚡</span>
          </div>
          <span className="font-bold text-2xl bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
            HDRit
          </span>
        </a>
        <div className="flex items-center gap-6">
          <a href="/pricing" className="text-gray-400 hover:text-white transition font-medium">
            Pricing
          </a>
          <a href="/dashboard" className="text-gray-400 hover:text-white transition font-medium">
            Dashboard
          </a>
          <a
            href="/dashboard"
            className="px-5 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold rounded-xl hover:from-cyan-400 hover:to-blue-400 transition-all shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40"
          >
            Sign In
          </a>
        </div>
      </nav>

      {/* Hero */}
      <div className="text-center mb-16">
        <h1 className="text-5xl font-bold mb-4">About HDRit</h1>
        <p className="text-xl text-gray-400">
          Born from friendship, fueled by frustration, built for photographers.
        </p>
      </div>

      {/* Origin Story */}
      <section className="mb-16">
        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl p-8 border border-gray-700">
          <h2 className="text-2xl font-bold mb-6 text-cyan-400">The Origin Story</h2>

          <div className="prose prose-invert max-w-none space-y-4 text-gray-300 leading-relaxed">
            <p>
              It was one of those winter nights at Tall Timbers where the cold outside makes the ideas inside burn brighter.
              Three friends—armed with nothing but laptops, too many MGDs, and an unreasonable amount of
              opinions about real estate photography—found themselves asking a dangerous question:
            </p>

            <p className="text-xl text-white font-medium italic border-l-4 border-cyan-400 pl-4 my-6">
              "Why does professional HDR editing still feel like it belongs in 2015?"
            </p>

            <p>
              What started as casual griping quickly evolved into furious whiteboarding. By 2 AM,
              the pizza was cold, the MGDs were warm, but the idea was red hot.
            </p>

            <p>
              <strong className="text-white">Jeff</strong>, the fourth member of the crew, contributed in his own way—by
              drinking and smoking in the corner, occasionally nodding in agreement, and providing crucial moral support.
              Every great team needs someone to keep the vibes right.
            </p>

            <p>
              <strong className="text-white">James</strong> saw the technical void—a space where AI could do the heavy
              lifting that photographers shouldn't have to. While others were still debating presets,
              he was already architecting neural pipelines and muttering about "perceptual tone mapping"
              like it was perfectly normal dinner conversation. Spoiler: it wasn't. But he was right.
            </p>

            <p>
              <strong className="text-white">Austin</strong>, the photographer of the trio with an eye sharper than
              a Phase One sensor, knew exactly what "good" looked like—because he'd spent years achieving it
              the hard way. Every slider, every parameter, every subtle shadow lift in HDRit? That's Austin's
              obsessive attention to detail, distilled into code. He's the reason your photos don't look like
              they were edited by a robot having an existential crisis.
            </p>

            <p>
              <strong className="text-white">Daniele</strong> brought the secret sauce—years of photo editing mastery
              that most people don't even know exists. Color science? Luminosity masking? The dark arts of
              making a $400,000 listing look like a $4 million estate? That's Daniele's playground.
              He speaks fluent LAB color space and dreams in histograms.
            </p>
          </div>
        </div>
      </section>

      {/* The Mission */}
      <section className="mb-16">
        <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 rounded-2xl p-8 border border-blue-700/50">
          <h2 className="text-2xl font-bold mb-6 text-blue-400">The Mission</h2>

          <div className="space-y-4 text-gray-300 leading-relaxed">
            <p>
              Real estate photographers are artists trapped in a world that treats them like button-pushers.
              They spend hours—sometimes days—editing brackets that should take minutes. They pay subscription
              fees for software that hasn't innovated since the Obama administration. They deserve better.
            </p>

            <p>
              HDRit exists to give photographers their time back. To make professional-grade HDR editing
              accessible to everyone—from the solo agent shooting their own listings to the high-volume
              production houses processing hundreds of properties a week.
            </p>

            <p className="text-lg text-white font-medium">
              We're not just building software. We're building the tool we wish existed when we started.
            </p>
          </div>
        </div>
      </section>

      {/* The Team */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold mb-8 text-center">The Team</h2>

        <div className="grid md:grid-cols-4 gap-6">
          {/* James */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 hover:border-cyan-400/50 transition">
            <div className="text-4xl mb-4">🧠</div>
            <h3 className="text-xl font-bold mb-2">James</h3>
            <p className="text-cyan-400 text-sm mb-3">The Architect</p>
            <p className="text-gray-400 text-sm">
              Turns MGDs into algorithms. Believes every problem is a software problem.
              Has strong opinions about color spaces and will share them unprompted.
            </p>
          </div>

          {/* Austin */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 hover:border-cyan-400/50 transition">
            <div className="text-4xl mb-4">📷</div>
            <h3 className="text-xl font-bold mb-2">Austin</h3>
            <p className="text-cyan-400 text-sm mb-3">The Eye</p>
            <p className="text-gray-400 text-sm">
              Master photographer who can spot a blown highlight from across the room.
              His parameters are the reason your photos look like photos, not fever dreams.
            </p>
          </div>

          {/* Daniele */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 hover:border-cyan-400/50 transition">
            <div className="text-4xl mb-4">🎨</div>
            <h3 className="text-xl font-bold mb-2">Daniele</h3>
            <p className="text-cyan-400 text-sm mb-3">The Alchemist</p>
            <p className="text-gray-400 text-sm">
              Photo editing wizard who treats pixels like paint. Knows secrets about color grading
              that Photoshop engineers don't. Probably a wizard. Unconfirmed.
            </p>
          </div>

          {/* Jeff */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 hover:border-cyan-400/50 transition">
            <div className="text-4xl mb-4">🍺</div>
            <h3 className="text-xl font-bold mb-2">Jeff</h3>
            <p className="text-cyan-400 text-sm mb-3">The Vibe</p>
            <p className="text-gray-400 text-sm">
              Professional drinker and smoker. Contributed zero lines of code but 100% of the morale.
              Still not entirely sure what HDR stands for.
            </p>
          </div>
        </div>
      </section>

      {/* Fun Facts */}
      <section className="mb-16">
        <div className="bg-gray-900/50 rounded-2xl p-8 border border-gray-800">
          <h2 className="text-2xl font-bold mb-6 text-center">By The Numbers</h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div>
              <div className="text-3xl font-bold text-cyan-400">1</div>
              <div className="text-gray-400 text-sm">Winter night</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-cyan-400">4</div>
              <div className="text-gray-400 text-sm">Stubborn friends</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-cyan-400">∞</div>
              <div className="text-gray-400 text-sm">MGDs consumed</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-cyan-400">0</div>
              <div className="text-gray-400 text-sm">Regrets</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="text-center mb-16">
        <h2 className="text-2xl font-bold mb-4 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">Ready to HDRit?</h2>
        <p className="text-gray-400 mb-6">
          Join thousands of photographers who've already made the switch.
        </p>
        <div className="flex justify-center gap-4">
          <a
            href="/"
            className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold rounded-xl hover:from-cyan-400 hover:to-blue-400 transition-all shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40"
          >
            Try It Free
          </a>
          <a
            href="/pricing"
            className="px-6 py-3 bg-gray-800/50 text-white font-semibold rounded-xl hover:bg-gray-800 transition border border-gray-700"
          >
            View Pricing
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-12 pt-8 border-t border-gray-800 text-center text-gray-500 text-sm">
        <p>HDRit • Made by <a href="https://linky.my" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 transition">Virul</a></p>
        <p className="mt-2 text-xs text-gray-600">
          © 2026 HDRit. All rights reserved.
        </p>
      </footer>
      </div>
    </main>
  )
}
