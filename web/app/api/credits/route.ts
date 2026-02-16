/**
 * Credits API Routes
 * Handles credit balance, purchases (top-ups), and usage tracking
 */

import { auth } from '@clerk/nextjs/server'
import { NextRequest, NextResponse } from 'next/server'
import { kv } from '@vercel/kv'

// Credit pack pricing
const CREDIT_PACKS = {
  small: { credits: 50, price: 30 },      // $0.60/photo
  medium: { credits: 200, price: 100 },   // $0.50/photo
  large: { credits: 500, price: 225 },    // $0.45/photo
  xl: { credits: 1000, price: 400 },      // $0.40/photo
} as const

// Default user data for new users
const DEFAULT_USER = {
  plan: 'free',
  credits: 10,
  creditsUsed: 0,
  status: 'active',
}

// GET /api/credits - Get user's credit balance
export async function GET() {
  try {
    const { userId } = await auth()

    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // Get user data from KV store
    let userData = await kv.hgetall(`user:${userId}`)

    // If new user, create default profile
    if (!userData || Object.keys(userData).length === 0) {
      userData = { ...DEFAULT_USER, createdAt: new Date().toISOString() }
      await kv.hset(`user:${userId}`, userData)
    }

    return NextResponse.json({
      success: true,
      data: {
        plan: userData.plan || 'free',
        credits: userData.credits || 10,
        creditsUsed: userData.creditsUsed || 0,
        status: userData.status || 'active',
        subscribedAt: userData.subscribedAt,
        currentPeriodEnd: userData.currentPeriodEnd,
      },
    })
  } catch (error: any) {
    console.error('Credits fetch error:', error)
    return NextResponse.json({ success: false, error: error.message }, { status: 500 })
  }
}

// POST /api/credits - Use a credit or purchase pack
export async function POST(request: NextRequest) {
  try {
    const { userId } = await auth()

    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const body = await request.json()
    const { action, packId } = body

    // Get current user data
    let userData = await kv.hgetall(`user:${userId}`)
    if (!userData || Object.keys(userData).length === 0) {
      userData = { ...DEFAULT_USER, createdAt: new Date().toISOString() }
      await kv.hset(`user:${userId}`, userData)
    }

    const currentCredits = (userData.credits as number) || 0
    const creditsUsed = (userData.creditsUsed as number) || 0

    if (action === 'use') {
      // Deduct a credit for processing
      if (currentCredits <= 0) {
        return NextResponse.json({
          success: false,
          error: 'No credits remaining. Please upgrade your plan or purchase more credits.',
        }, { status: 402 })
      }

      await kv.hset(`user:${userId}`, {
        credits: currentCredits - 1,
        creditsUsed: creditsUsed + 1,
        lastUsedAt: new Date().toISOString(),
      })

      // Log the usage
      await kv.lpush(`user:${userId}:logs`, JSON.stringify({
        action: 'use',
        amount: -1,
        balance: currentCredits - 1,
        timestamp: new Date().toISOString(),
      }))

      return NextResponse.json({
        success: true,
        data: {
          credits: currentCredits - 1,
          creditsUsed: creditsUsed + 1,
        },
      })
    }

    if (action === 'purchase' && packId) {
      // Purchase credit pack
      const pack = CREDIT_PACKS[packId as keyof typeof CREDIT_PACKS]

      if (!pack) {
        return NextResponse.json({
          success: false,
          error: 'Invalid pack ID',
          availablePacks: Object.entries(CREDIT_PACKS).map(([id, p]) => ({
            id,
            credits: p.credits,
            price: p.price,
            pricePerPhoto: `$${(p.price / p.credits).toFixed(2)}`,
          })),
        }, { status: 400 })
      }

      // In production, this would redirect to Stripe checkout
      // For now, simulate successful purchase
      const newBalance = currentCredits + pack.credits

      await kv.hset(`user:${userId}`, {
        credits: newBalance,
        lastPurchaseAt: new Date().toISOString(),
      })

      // Log the purchase
      await kv.lpush(`user:${userId}:logs`, JSON.stringify({
        action: 'purchase',
        packId,
        amount: pack.credits,
        price: pack.price,
        balance: newBalance,
        timestamp: new Date().toISOString(),
      }))

      return NextResponse.json({
        success: true,
        data: {
          creditsAdded: pack.credits,
          amountCharged: pack.price,
          newBalance,
        },
        message: `Successfully added ${pack.credits} credits to your account!`,
      })
    }

    return NextResponse.json({ success: false, error: 'Invalid action' }, { status: 400 })
  } catch (error: any) {
    console.error('Credits update error:', error)
    return NextResponse.json({ success: false, error: error.message }, { status: 500 })
  }
}

// GET available credit packs
export async function OPTIONS() {
  return NextResponse.json({
    success: true,
    data: {
      packs: Object.entries(CREDIT_PACKS).map(([id, pack]) => ({
        id,
        credits: pack.credits,
        price: pack.price,
        pricePerPhoto: `$${(pack.price / pack.credits).toFixed(2)}`,
        savings: id === 'small' ? null : `${Math.round((1 - (pack.price / pack.credits) / 0.6) * 100)}% off`,
      })),
    },
  })
}
