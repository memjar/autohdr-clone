'use client'

// SVG Icons
const Icons = {
  brain: (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  ),
  camera: (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
  palette: (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
    </svg>
  ),
  glass: (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  ),
}

export default function About() {
  return (
    <main className="min-h-screen bg-[#0a0a0f]">
      {/* Subtle background gradient */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 py-8 pb-24">
      {/* Navigation */}
      <nav className="flex justify-between items-center mb-12 pb-6 border-b border-white/5">
        <a href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <span className="text-white font-bold text-sm">H</span>
          </div>
          <span className="font-semibold text-white text-lg">HDRit</span>
        </a>
        <div className="flex items-center gap-6">
          <a href="/pricing" className="text-sm text-gray-400 hover:text-white transition">
            Pricing
          </a>
          <a href="/dashboard" className="text-sm text-gray-400 hover:text-white transition">
            Dashboard
          </a>
          <a
            href="/dashboard"
            className="px-4 py-2 text-sm font-medium text-white bg-white/10 hover:bg-white/15 rounded-lg transition"
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
          <h2 className="text-2xl font-bold mb-6 text-blue-400">The Origin Story</h2>

          <div className="prose prose-invert max-w-none space-y-4 text-gray-300 leading-relaxed">
            <p>
              It was one of those winter nights at Tall Timbers where the cold outside makes the ideas inside burn brighter.
              Three friends—armed with nothing but laptops, too many MGDs, and an unreasonable amount of
              opinions about real estate photography—found themselves asking a dangerous question:
            </p>

            <p className="text-xl text-white font-medium italic border-l-4 border-blue-400 pl-4 my-6">
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
          <div className="bg-white/5 rounded-xl p-6 border border-white/10 hover:border-blue-400/50 transition">
            <div className="text-blue-400 mb-4">{Icons.brain}</div>
            <h3 className="text-xl font-bold mb-2 text-white">James</h3>
            <p className="text-blue-400 text-sm mb-3">The Architect</p>
            <p className="text-gray-400 text-sm">
              Turns MGDs into algorithms. Believes every problem is a software problem.
              Has strong opinions about color spaces and will share them unprompted.
            </p>
          </div>

          {/* Austin */}
          <div className="bg-white/5 rounded-xl p-6 border border-white/10 hover:border-blue-400/50 transition">
            <div className="text-blue-400 mb-4">{Icons.camera}</div>
            <h3 className="text-xl font-bold mb-2 text-white">Austin</h3>
            <p className="text-blue-400 text-sm mb-3">The Eye</p>
            <p className="text-gray-400 text-sm">
              Master photographer who can spot a blown highlight from across the room.
              His parameters are the reason your photos look like photos, not fever dreams.
            </p>
          </div>

          {/* Daniele */}
          <div className="bg-white/5 rounded-xl p-6 border border-white/10 hover:border-blue-400/50 transition">
            <div className="text-blue-400 mb-4">{Icons.palette}</div>
            <h3 className="text-xl font-bold mb-2 text-white">Daniele</h3>
            <p className="text-blue-400 text-sm mb-3">The Alchemist</p>
            <p className="text-gray-400 text-sm">
              Photo editing wizard who treats pixels like paint. Knows secrets about color grading
              that Photoshop engineers don't. Probably a wizard. Unconfirmed.
            </p>
          </div>

          {/* Jeff */}
          <div className="bg-white/5 rounded-xl p-6 border border-white/10 hover:border-blue-400/50 transition">
            <div className="text-blue-400 mb-4">{Icons.glass}</div>
            <h3 className="text-xl font-bold mb-2 text-white">Jeff</h3>
            <p className="text-blue-400 text-sm mb-3">The Vibe</p>
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
              <div className="text-3xl font-bold text-blue-400">1</div>
              <div className="text-gray-400 text-sm">Winter night</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-400">4</div>
              <div className="text-gray-400 text-sm">Stubborn friends</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-400">∞</div>
              <div className="text-gray-400 text-sm">MGDs consumed</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-400">0</div>
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
            className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium rounded-lg hover:from-blue-400 hover:to-purple-500 transition-all"
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
        <p>HDRit • Made by <a href="https://linky.my" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 transition">Virul</a></p>
        <p className="mt-2 text-xs text-gray-600">
          © 2026 HDRit. All rights reserved.
        </p>
      </footer>
      </div>
    </main>
  )
}
