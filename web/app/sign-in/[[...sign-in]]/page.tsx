import { SignIn } from '@clerk/nextjs'

export default function SignInPage() {
  return (
    <div className="min-h-screen bg-black flex items-center justify-center px-4">
      <SignIn
        appearance={{
          elements: {
            rootBox: 'mx-auto',
            card: 'bg-zinc-900 border border-white/10',
            headerTitle: 'text-white',
            headerSubtitle: 'text-gray-400',
            socialButtonsBlockButton: 'bg-white/10 border-white/20 text-white hover:bg-white/20',
            formFieldLabel: 'text-gray-300',
            formFieldInput: 'bg-zinc-800 border-white/10 text-white',
            footerActionLink: 'text-blue-400 hover:text-blue-300',
            identityPreviewText: 'text-white',
            identityPreviewEditButton: 'text-blue-400',
          },
        }}
      />
    </div>
  )
}
